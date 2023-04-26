from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.functional import normalize
import wandb
from transformers import AdamW, AutoTokenizer, AutoModel
import torchmetrics

from losses import SupervisedContrastiveLoss
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score


class Encoder(object):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, sentences: List, normalize_embeddings=True, device=None) -> torch.Tensor:
        encoded_input = self.tokenizer.batch_encode_plus(sentences, padding=True, truncation=True, return_attention_mask=True, return_tensors='pt', return_token_type_ids=False)
        attention_mask = encoded_input["attention_mask"].to(device=device)
        model_output = self.model(encoded_input["input_ids"].to(device=device), attention_mask)
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        embeddings = normalize(sentence_embeddings, p=2, dim=1) if normalize_embeddings else sentence_embeddings
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class SentenceTransformersModel(pl.LightningModule):
    def __init__(self, n_classes: int, lr: float, temperature: float, alpha: float):
        super().__init__()
        self.lr = lr
        self.temperature = temperature
        self.alpha = alpha
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.lm = Encoder(self.model, self.tokenizer)
        # self.classifier = nn.Linear(384, n_classes, device=self.device, dtype=torch.float32)
        # self.loss = nn.CrossEntropyLoss()
        self.num_clusters = 20
        self.clustering_model = DBSCAN(eps=0.3, min_samples=3, metric="cosine")
        self.accuracy = torchmetrics.classification.Accuracy(num_classes=n_classes, task="multiclass").to(self.device)
        self.preci = torchmetrics.classification.Precision(num_classes=n_classes, task="multiclass").to(self.device)
        self.recall = torchmetrics.classification.Recall(num_classes=n_classes, task="multiclass").to(self.device)
        self.f1 = torchmetrics.classification.F1Score(num_classes=n_classes, task="multiclass").to(self.device)
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=self.temperature)
        self.column = ["sentences", "predictions", "labels"]
        self.test_table = wandb.Table(columns=["sentences", "predictions", "labels"])
        self.validation_table = wandb.Table(columns=["sentences", "predictions", "labels"])
        self.save_hyperparameters()

    def forward(self, sentences: list, labels=None, is_training=True, is_contrastive=True, mode="train"):
        labels_on_cpu = labels.cpu().numpy().flatten()
        sentence_embeddings = self.lm.encode(sentences, normalize_embeddings=True, device=self.device)
        embeddings_on_cpu = sentence_embeddings.detach().cpu().numpy()
        cluster_labels = self.clustering_model.fit_predict(embeddings_on_cpu)
        self.log(f"{mode}/num. clusters", int(len(set(cluster_labels))))
        # predictions = self.infer_labels_from_clusters(cluster_labels, labels_on_cpu)

        if is_contrastive and mode in ["train", "valid"]:
            nmi = adjusted_mutual_info_score(cluster_labels, labels_on_cpu)
            self.log(f"{mode}/nmi", nmi)
            ari = adjusted_rand_score(cluster_labels, labels_on_cpu)
            self.log(f"{mode}/ari", ari)
            multiview_sentences, multiview_labels = self.get_multiview_batch(sentence_embeddings, labels.flatten())
            contrastive_loss = self.contrastive_loss(multiview_sentences, multiview_labels)
            self.log(f"{mode}/contrastive loss", contrastive_loss)
            return self.alpha*contrastive_loss
        else:
            # cluster_loss = silhouette_score(sentence_embeddings.cpu().numpy(), labels_np)
            # self.log(f"{mode}/silhouette_score", cluster_loss)
            # loss = self.loss(sentence_embeddings, labels.long())
            # self.log(f"{mode}/ce loss", loss)
            return 0

    def infer_labels_from_clusters(self, cluster_labels, true_labels):
        for i in set(cluster_labels):
            index_for_cluster_i = np.argwhere(cluster_labels == i).flatten()
            true_labels_for_cluster_i = true_labels[index_for_cluster_i].flatten()
            unique_counts = np.unique(true_labels_for_cluster_i, return_counts=True)
            max_label_count_for_cluster_i = max(unique_counts[1])
            label_for_cluster_i = np.random.choice(np.argwhere(unique_counts[0] == max_label_count_for_cluster_i).flatten())
        return []

    def get_multiview_batch(self, features, labels, dummy=False):
        # no augmentation
        if dummy:
            contrastive_features = features[:, None, :]
            contrastive_labels = labels
        else:
            multiview_shape = (
                int(features.shape[0] / (2 + 1)),
                2 + 1,
                features.shape[-1]
            )
            contrastive_features = features.reshape(multiview_shape)
            contrastive_labels = labels[:int(features.shape[0]/(2+1))]
        return contrastive_features, contrastive_labels

    def augment(self, batch, num_return_sequences: int = 2):
        augmented_text = []
        sentences, labels = batch["text"], batch["label"]
        sentences = list(sentences)
        for n in range(num_return_sequences+1):
            augmented_text.extend(sentences)
        # label need to repeat n times + the original copy
        augmented_label = labels.repeat(num_return_sequences + 1, 1)
        return augmented_text, augmented_label

    def training_step(self, batch, batch_idx):
        sentences, labels = self.augment(batch)
        loss = self.forward(sentences, labels, is_contrastive=True, mode="train")
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "labels": labels.flatten()}

    def validation_step(self, batch, batch_idx):
        sentences, labels = self.augment(batch)
        loss = self.forward(sentences, labels, is_training=False, is_contrastive=True, mode="valid")
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "labels": labels.flatten()}

    def test_step(self, batch, batch_idx):
        sentences, labels = batch
        loss = self.forward(sentences, labels, is_training=False, is_contrastive=False, mode="test")
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "labels": labels.flatten()}

    def evaluate(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"]:
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                predictions.append(out_predictions)
        labels = torch.stack(labels).flatten().int()
        predictions = torch.stack(predictions)
        acc = self.accuracy(predictions, labels)
        preci = self.preci(predictions, labels)
        recall = self.recall(predictions, labels)
        f1 = self.f1(predictions, labels)
        return acc, preci, recall, f1

    def training_epoch_end(self, outputs):
        # acc, preci, recall, f1 = self.evaluate(outputs)
        # self.log("train/acc", acc, prog_bar=True, logger=True, on_epoch=True)
        # self.log("train/preci", preci, prog_bar=True, logger=True, on_epoch=True)
        # self.log("train/recall", recall, prog_bar=True, logger=True, on_epoch=True)
        # self.log("train/f1", f1, prog_bar=True, logger=True, on_epoch=True)
        pass

    def validation_epoch_end(self, outputs):
        # acc, preci, recall, f1 = self.evaluate(outputs)
        # self.log("validation/acc", acc, prog_bar=True, logger=True)
        # self.log("validation/preci", preci, prog_bar=True, logger=True)
        # self.log("validation/recall", recall, prog_bar=True, logger=True)
        # self.log("validation/f1", f1, prog_bar=True, logger=True)
        pass

    def test_epoch_end(self, outputs):
        # acc, preci, recall, f1 = self.evaluate(outputs)
        # self.log("test/acc", acc, prog_bar=True, logger=True)
        # self.log("test/preci", preci, prog_bar=True, logger=True)
        # self.log("test/recall", recall, prog_bar=True, logger=True)
        # self.log("test/f1", f1, prog_bar=True, logger=True)
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.n_warmup_steps,
        #     num_training_steps=self.n_training_steps
        # )
        return dict(
            optimizer=optimizer#,
            # lr_scheduler=dict(
            #     scheduler=scheduler,
            #     interval='step'
            # )
        )
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
import hdbscan
from sklearn.metrics import silhouette_score
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
    def __init__(self, n_classes: int, lr: float, temperature: float, alpha: float, num_augmentation: int = 2, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.num_augmentation = num_augmentation
        self.lr = lr
        self.temperature = temperature
        self.alpha = alpha
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.lm = Encoder(self.model, self.tokenizer)
        # self.classifier = nn.Linear(384, n_classes, device=self.device, dtype=torch.float32)
        # self.loss = nn.CrossEntropyLoss()
        self.num_clusters = n_classes
        self.clustering_model = self.instantiate_clustering_model("hdbscan")
        self.accuracy = torchmetrics.classification.Accuracy(num_classes=n_classes, task="multiclass").to(self.device)
        self.preci = torchmetrics.classification.Precision(num_classes=n_classes, task="multiclass").to(self.device)
        self.recall = torchmetrics.classification.Recall(num_classes=n_classes, task="multiclass").to(self.device)
        self.f1 = torchmetrics.classification.F1Score(num_classes=n_classes, task="multiclass").to(self.device)
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=self.temperature)
        self.column = ["sentences", "predictions", "labels"]
        self.test_table = wandb.Table(columns=["sentences", "predictions", "labels"])
        self.validation_table = wandb.Table(columns=["sentences", "predictions", "labels"])
        self.save_hyperparameters()

    def instantiate_clustering_model(self, name: str):
        if name == "hdbscan":
            min_cluster_size = self.kwargs.get('min_cluster_size', 10)
            clustering_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        elif name == "dbscan":
            eps = self.kwargs.get('eps', 0.3)
            min_samples = self.kwargs.get('min_samples', 3)
            clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        else:
            raise NotImplementedError
        return clustering_model

    def forward(self, sentences: list, labels=None, is_training=True, is_contrastive=True, mode="train"):
        labels_on_cpu = labels.cpu().numpy().flatten()
        sentence_embeddings = self.lm.encode(sentences, normalize_embeddings=True, device=self.device)
        embeddings_on_cpu = sentence_embeddings.detach().cpu().numpy()
        cluster_labels = self.clustering_model.fit_predict(embeddings_on_cpu)
        num_clusters = len(set(cluster_labels))
        self.log(f"{mode}/num. clusters", num_clusters)
        predictions = self.get_predicted_cluster_labels(cluster_labels, labels_on_cpu, embeddings_on_cpu)

        if is_contrastive and mode in ["train", "valid"]:
            nmi = adjusted_mutual_info_score(cluster_labels, labels_on_cpu)
            self.log(f"{mode}/nmi", nmi)
            ari = adjusted_rand_score(cluster_labels, labels_on_cpu)
            self.log(f"{mode}/ari", ari)
            multiview_sentences, multiview_labels = self.get_multiview_batch(sentence_embeddings, labels.flatten())
            contrastive_loss = self.contrastive_loss(multiview_sentences, multiview_labels)
            self.log(f"{mode}/contrastive loss", contrastive_loss)
            return self.alpha*contrastive_loss, predictions
        else:
            return 0, predictions

    def get_multiview_batch(self, features, labels, dummy=False):
        # no augmentation
        if dummy:
            contrastive_features = features[:, None, :]
            contrastive_labels = labels
        else:
            multiview_shape = (
                int(features.shape[0] / (self.num_augmentation + 1)),
                self.num_augmentation + 1,
                features.shape[-1]
            )
            contrastive_features = features.reshape(multiview_shape)
            contrastive_labels = labels[:int(features.shape[0]/(self.num_augmentation+1))]
        return contrastive_features, contrastive_labels

    def augment(self, batch):
        augmented_text = []
        sentences, labels = batch["text"], batch["label"]
        sentences = list(sentences)
        for n in range(self.num_augmentation+1):
            augmented_text.extend(sentences)
        # label need to repeat n times + the original copy
        augmented_label = labels.repeat(self.num_augmentation + 1, 1)
        return augmented_text, augmented_label

    def training_step(self, batch, batch_idx):
        sentences, labels = self.augment(batch)
        loss, predictions = self.forward(sentences, labels, is_contrastive=True, mode="train")
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": predictions, "labels": labels.flatten()}

    def validation_step(self, batch, batch_idx):
        sentences, labels = self.augment(batch)
        loss, predictions = self.forward(sentences, labels, is_training=False, is_contrastive=True, mode="valid")
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": predictions, "labels": labels.flatten()}

    def test_step(self, batch, batch_idx):
        sentences, labels = batch
        loss, predictions = self.forward(sentences, labels, is_training=False, is_contrastive=False, mode="test")
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": predictions, "labels": labels.flatten()}

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
        acc, preci, recall, f1 = self.evaluate(outputs)
        self.log("train/acc", acc, prog_bar=True, logger=True, on_epoch=True)
        self.log("train/preci", preci, prog_bar=True, logger=True, on_epoch=True)
        self.log("train/recall", recall, prog_bar=True, logger=True, on_epoch=True)
        self.log("train/f1", f1, prog_bar=True, logger=True, on_epoch=True)

    def validation_epoch_end(self, outputs):
        acc, preci, recall, f1 = self.evaluate(outputs)
        self.log("validation/acc", acc, prog_bar=True, logger=True)
        self.log("validation/preci", preci, prog_bar=True, logger=True)
        self.log("validation/recall", recall, prog_bar=True, logger=True)
        self.log("validation/f1", f1, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):
        acc, preci, recall, f1 = self.evaluate(outputs)
        self.log("test/acc", acc, prog_bar=True, logger=True)
        self.log("test/preci", preci, prog_bar=True, logger=True)
        self.log("test/recall", recall, prog_bar=True, logger=True)
        self.log("test/f1", f1, prog_bar=True, logger=True)

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

    @staticmethod
    def get_silhouette_score(label, cluster_indices, cluster_labels, true_labels, embeddings):
        label_indices = [i for i in cluster_indices if true_labels[i] == label]
        label_embeddings = embeddings[label_indices]

        if len(label_embeddings) < 2:
            return -1

        return silhouette_score(embeddings, cluster_labels[label_indices], metric='cosine')

    @staticmethod
    def assign_label_to_cluster(cluster_labels, true_labels, embeddings):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
        label_counts = {}
        for idx in cluster_indices:
            label = true_labels[idx]
            label_counts[label] = label_counts.get(label, 0) + 1

        competing_labels = [label for label, count in label_counts.items() if count == max(label_counts.values())]

        if len(competing_labels) == 1:
            return competing_labels[0]

        best_label = competing_labels[0]
        best_silhouette_score = self.get_silhouette_score(best_label, cluster_indices, cluster_labels, embeddings)

        for label in competing_labels[1:]:
            current_silhouette_score = self.get_silhouette_score(label, cluster_indices, cluster_labels, embeddings)

            if current_silhouette_score > best_silhouette_score:
                best_label = label
                best_silhouette_score = current_silhouette_score

        return best_label

    def get_predicted_cluster_labels(self, predicted_labels, true_labels, embeddings):
        # Assign labels to clusters
        unique_clusters = set(predicted_labels)
        cluster_labels = {}

        for cluster in unique_clusters:
            if cluster == -1:  # Noise
                continue
            cluster_label = self.assign_label_to_cluster(predicted_labels, true_labels, embeddings)
            cluster_labels[cluster] = cluster_label
        predictions = [cluster_labels.get(predicted_label) for predicted_label in predicted_labels]
        return predictions

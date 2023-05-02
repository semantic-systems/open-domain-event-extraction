import json
from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.nn.functional import normalize
import wandb
from transformers import AdamW, AutoTokenizer, AutoModel
import torchmetrics
from sklearn.metrics import pairwise_distances
import plotly.express as px
from sklearn.decomposition import PCA
import mpld3

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
        self.clustering_model = self.instantiate_clustering_model("kmeans")
        self.accuracy = torchmetrics.classification.Accuracy(num_classes=n_classes, task="multiclass", average='macro').to(self.device)
        self.preci = torchmetrics.classification.Precision(num_classes=n_classes, task="multiclass", average='macro').to(self.device)
        self.recall = torchmetrics.classification.Recall(num_classes=n_classes, task="multiclass", average='macro').to(self.device)
        self.f1 = torchmetrics.classification.F1Score(num_classes=n_classes, task="multiclass", average='macro').to(self.device)
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=self.temperature)
        self.column = ["sentences", "predictions", "labels"]
        self.test_table = wandb.Table(columns=["sentences", "predictions", "labels"])
        self.validation_table = wandb.Table(columns=["sentences", "predictions", "labels"])
        self.test_embedding_table = wandb.Table(columns=["clustering", "classification"])
        self.validation_embedding_table = wandb.Table(columns=["clustering", "classification"])
        self.index_label_map = None
        self.pca = PCA(n_components=2)
        self.save_hyperparameters()

    def reduce_with_PCA(self, features):
        return self.pca.fit_transform(features)

    def instantiate_clustering_model(self, name: str):
        if name == "hdbscan":
            min_cluster_size = self.kwargs.get('min_cluster_size', 10)
            clustering_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        elif name == "dbscan":
            eps = self.kwargs.get('eps', 0.3)
            min_samples = self.kwargs.get('min_samples', 3)
            clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        elif name == "kmeans":
            clustering_model = KMeans(n_clusters=self.num_clusters, n_init=10)
        else:
            raise NotImplementedError
        return clustering_model

    def forward(self, sentences: list, labels=None, is_training=True, is_contrastive=True, mode="train"):
        labels_on_cpu = labels.cpu().numpy().flatten()
        sentence_embeddings = self.lm.encode(sentences, normalize_embeddings=False, device=self.device)
        embeddings_on_cpu = sentence_embeddings.detach().cpu().numpy()
        cluster_labels = self.clustering_model.fit_predict(embeddings_on_cpu)
        num_clusters = len(set(cluster_labels))
        self.log(f"{mode}/num. clusters", num_clusters)
        predictions = self.get_predicted_cluster_labels(cluster_labels, labels_on_cpu, embeddings_on_cpu)
        self.get_pca_plot(embeddings_on_cpu, predictions, labels_on_cpu, sentences, cluster_labels)

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
            self.test_embedding_table.add_data([wandb.Html("fig_cluster.html")], [wandb.Html("fig_cls.html")])
            return 0, predictions

    def get_pca_plot(self, embeddings, predictions, labels, sentences, cluster_labels):
        reduced_embeddings = self.reduce_with_PCA(embeddings)
        df = pd.DataFrame({"PC 1": reduced_embeddings[:, 0], "PC 2": reduced_embeddings[:, 1], "cluster label": cluster_labels,
                           "predictions": predictions, "labels": labels, "sentences": sentences})

        fig_cls = px.scatter(df, x="PC 1", y="PC 2", color="labels",
                             hover_data=['sentences'])  # ,width=1000, height=700)
        fig_cls.update_traces(marker_size=10)
        fig_cluster = px.scatter(df, x="PC 1", y="PC 2", color="cluster label",
                                 hover_data=['sentences'])  # , width=1065, height=700)
        fig_cluster.update_traces(marker_size=10)

        fig_cls.update_layout(
            title="PCA Visualization with Classification Result",
            title_x=0.5,
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            font=dict(
                size=12
            ),
            hoverlabel=dict(
                font_size=12,
            ),
            margin=dict(l=15, r=15, t=15, b=15),
            paper_bgcolor="#E8E8DC"
        )
        fig_cluster.update_layout(
            title="PCA Visualization with Clustering Result",
            title_x=0.5,
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            font=dict(
                size=12
            ),
            hoverlabel=dict(
                font_size=12,
            ),
            margin=dict(l=15, r=15, t=15, b=15),
            paper_bgcolor="#E8E8DC"
        )
        fig_cls.write_html("fig_cls.html")
        fig_cluster.write_html("fig_cluster.html")
        # return fig_cls, fig_cluster

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
        predictions_placeholder = torch.tensor(predictions, device=self.device, dtype=torch.int) if predictions is not None else predictions
        return {"loss": loss, "predictions": predictions_placeholder, "labels": labels.flatten()}

    def validation_step(self, batch, batch_idx):
        sentences, labels = self.augment(batch)
        loss, predictions = self.forward(sentences, labels, is_training=False, is_contrastive=True, mode="valid")
        self.log("val_loss", loss, prog_bar=True, logger=True)
        predictions_placeholder = torch.tensor(predictions, device=self.device, dtype=torch.int) if predictions is not None else predictions
        return {"loss": loss, "predictions": predictions_placeholder, "labels": labels.flatten()}

    def test_step(self, batch, batch_idx):
        sentences, labels = batch["text"], batch["label"]
        loss, predictions = self.forward(sentences, labels, is_training=False, is_contrastive=False, mode="test")
        self.log("test_loss", loss, prog_bar=True, logger=True)
        predictions_placeholder = torch.tensor(predictions, device=self.device, dtype=torch.int) if predictions is not None else predictions
        data = [[s, pred, label] for s, pred, label in
                list(zip(sentences, self.get_label_in_string(predictions_placeholder, len(sentences)),
                         self.get_label_in_string(labels.flatten())))]
        for step in data:
            self.test_table.add_data(*step)
        return {"loss": loss, "predictions": predictions_placeholder, "labels": labels.flatten()}

    def get_label_in_string(self, predictions, count = 0):
        if predictions is None:
            return ["oos"] * count
        if self.index_label_map is None:
            label_map_file = "./index_label_maps/emotion.json"
            with open(label_map_file, 'r') as f:
                self.index_label_map = json.load(f)
        return [self.index_label_map[str(pred.item())] for pred in predictions]

    def evaluate(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            if output["predictions"] is None:
                continue
            for out_labels in output["labels"]:
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                predictions.append(out_predictions)
        if len(labels) == 0:
            return 0, 0, 0, 0
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
        wandb.log({"test/result_table": self.test_table})
        wandb.log({"test/embedding_table": self.test_embedding_table})

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

    def get_predicted_cluster_labels(self, predicted_labels, true_labels, embeddings):
        # Assign labels to noise points
        # noise_indices = np.where(predicted_labels == -1)[0]
        # cluster_indices = np.where(predicted_labels != -1)[0]
        # if len(cluster_indices) == 0:
        #     return None
        # if len(cluster_indices) > 0:
        #     cluster_centroids = np.array(
        #         [np.mean(embeddings[predicted_labels == i], axis=0) for i in np.unique(predicted_labels) if i != -1])
        #     noise_points = embeddings[noise_indices]
        #     distances = pairwise_distances(noise_points, cluster_centroids)
        #     closest_cluster_indices = np.argmin(distances, axis=1)
        #     closest_cluster_labels = predicted_labels[cluster_indices][closest_cluster_indices]
        #     predicted_labels[noise_indices] = closest_cluster_labels

        # Assign labels to clusters
        unique_clusters = set(predicted_labels)
        cluster_labels = {}

        for cluster in unique_clusters:
            cluster_indices = [i for i, label in enumerate(predicted_labels) if label == cluster]
            cluster_label = self.assign_label_to_cluster(cluster_indices, predicted_labels, true_labels, embeddings)
            cluster_labels[cluster] = cluster_label
        predictions = [cluster_labels.get(predicted_label) for predicted_label in predicted_labels]
        return predictions

    def assign_label_to_cluster(self, cluster_indices, predicted_labels, true_labels, embeddings):
        label_counts = {}
        for idx in cluster_indices:
            label = true_labels[idx]
            label_counts[label] = label_counts.get(label, 0) + 1

        competing_labels = [label for label, count in label_counts.items() if count == max(label_counts.values())]

        if len(competing_labels) == 1:
            return competing_labels[0]

        best_label = competing_labels[0]
        best_distance = np.inf
        cluster_embeddings = embeddings[cluster_indices]
        cluster_centroid = np.mean(cluster_embeddings, axis=0)

        for label in competing_labels[1:]:
            label_indices = [i for i in range(len(true_labels)) if true_labels[i] == label]
            label_embeddings = embeddings[label_indices]
            label_centroid = np.mean(label_embeddings, axis=0)

            distance = pairwise_distances([cluster_centroid], [label_centroid], metric='cosine')[0][0]

            if distance < best_distance:
                best_label = label
                best_distance = distance
        return best_label
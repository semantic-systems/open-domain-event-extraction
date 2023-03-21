import pytorch_lightning as pl
from transformers import RobertaModel, AdamW, get_linear_schedule_with_warmup, RobertaTokenizer
import torch
import torch.nn as nn
import torchmetrics
import json
import wandb
from losses import HMLC


NUM_AUGMENTATION = 1


class MavenModel(pl.LightningModule):
    def __init__(self, n_classes: int, pretrained_model_name_or_path: str, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.lm = RobertaModel.from_pretrained(pretrained_model_name_or_path, return_dict=True)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.classifier = nn.Linear(self.lm.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.loss = nn.BCELoss()
        self.contrastive_loss = HMLC()
        self.auroc = torchmetrics.AUROC(task="multilabel", num_labels=169).to(self.device)
        self.accuracy = torchmetrics.classification.MultilabelAccuracy(num_labels=169).to(self.device)
        self.preci = torchmetrics.classification.MultilabelPrecision(num_labels=169).to(self.device)
        self.recall = torchmetrics.classification.MultilabelRecall(num_labels=169).to(self.device)
        self.f1 = torchmetrics.classification.MultilabelF1Score(num_labels=169).to(self.device)
        self.column = ["sentences", "predictions", "labels"]
        self.test_table = wandb.Table(columns=["sentences", "predictions", "labels"])
        self.validation_table = wandb.Table(columns=["sentences", "predictions", "labels"])
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None, is_training=True, is_contrastive=True):
        if is_contrastive and is_training:
            loss = 0
            logits = self.classifier(self.lm(input_ids=input_ids, attention_mask=attention_mask).pooler_output)
            output = torch.sigmoid(logits)
            multiview_sentences, multiview_labels = self.get_multiview_batch(logits, labels)
            contrastive_loss = self.contrastive_loss(multiview_sentences, multiview_labels)
            self.log("train/contrastive loss", contrastive_loss)
            if labels is not None:
                loss = self.loss(output, labels)
            return loss + contrastive_loss, output
        else:
            output = self.lm(input_ids, attention_mask=attention_mask)
            output = self.classifier(output.pooler_output)
            output = torch.sigmoid(output)
            loss = 0
            if labels is not None:
                loss = self.loss(output, labels)
            return loss, output

    def get_multiview_batch(self, features, labels, dummy=False):
        # no augmentation
        if dummy:
            contrastive_features = features[:, None, :]
            contrastive_labels = labels
        else:
            multiview_shape = (
                int(features.shape[0] / (NUM_AUGMENTATION + 1)),
                NUM_AUGMENTATION + 1,
                features.shape[-1]
            )
            contrastive_features = features.reshape(multiview_shape)
            contrastive_labels = labels[:int(features.shape[0]/(NUM_AUGMENTATION+1))]
        return contrastive_features, contrastive_labels

    def augment(self, batch, num_return_sequences: int = NUM_AUGMENTATION):
        augmented_text = []
        sentences, labels = batch
        sentences = list(sentences)
        for n in range(num_return_sequences+1):
            augmented_text.extend(sentences)
        # label need to repeat n times + the original copy
        augmented_label = labels.repeat(num_return_sequences + 1, 1)
        return augmented_text, augmented_label

    def training_step(self, batch, batch_idx):
        sentences, labels = self.augment(batch)
        features = self.tokenizer.batch_encode_plus(sentences, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt', return_token_type_ids=False)
        loss, outputs = self.forward(features["input_ids"].to(device=self.device), features["attention_mask"].to(device=self.device), labels.to(device=self.device))
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        sentences, labels = batch
        features = self.tokenizer.batch_encode_plus(sentences, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt', return_token_type_ids=False)
        loss, outputs = self.forward(features["input_ids"].to(device=self.device), features["attention_mask"].to(device=self.device), labels.to(device=self.device), is_training=False, is_contrastive=False)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        prediction_int = torch.as_tensor((outputs - 0.5) > 0, dtype=torch.int32)
        prediction_str = self.get_event_type(prediction_int)
        data = [[s, pred, label] for s, pred, label in list(zip(sentences, prediction_str, self.get_event_type(labels)))]
        for step in data:
            self.validation_table.add_data(*step)
        return {"loss": loss, "predictions": outputs, "labels": labels, "prediction_int": prediction_int}

    def test_step(self, batch, batch_idx):
        sentences, labels = batch
        features = self.tokenizer.batch_encode_plus(sentences, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt', return_token_type_ids=False)
        loss, outputs = self.forward(features["input_ids"].to(device=self.device), features["attention_mask"].to(device=self.device), labels.to(device=self.device), is_training=False, is_contrastive=False)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        prediction_int = torch.as_tensor((outputs - 0.5) > 0, dtype=torch.int32)
        prediction_str = self.get_event_type(prediction_int)
        data = [[s, pred, label] for s, pred, label in list(zip(sentences, prediction_str, self.get_event_type(labels)))]
        for step in data:
            self.test_table.add_data(*step)
        return {"loss": loss, "predictions": outputs, "labels": labels, "prediction_int": prediction_int}

    def evaluate(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"]:#.detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"]:#.detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        class_roc_auc = self.auroc(predictions, labels)
        acc = self.accuracy(predictions, labels)
        preci = self.preci(predictions, labels)
        recall = self.recall(predictions, labels)
        f1 = self.f1(predictions, labels)
        return class_roc_auc, acc, preci, recall, f1

    def training_epoch_end(self, outputs):
        class_roc_auc, acc, preci, recall, f1 = self.evaluate(outputs)
        self.log("train/acc", acc, prog_bar=True, logger=True, on_epoch=True)
        self.log("train/preci", preci, prog_bar=True, logger=True, on_epoch=True)
        self.log("train/recall", recall, prog_bar=True, logger=True, on_epoch=True)
        self.log("train/f1", f1, prog_bar=True, logger=True, on_epoch=True)
        self.log(f"train/roc_auc", class_roc_auc, prog_bar=True, logger=True, on_epoch=True)

    def validation_epoch_end(self, outputs):
        class_roc_auc, acc, preci, recall, f1 = self.evaluate(outputs)
        self.log("validation/acc", acc, prog_bar=True, logger=True)
        self.log("validation/preci", preci, prog_bar=True, logger=True)
        self.log("validation/recall", recall, prog_bar=True, logger=True)
        self.log("validation/f1", f1, prog_bar=True, logger=True)
        self.log(f"validation/roc_auc", class_roc_auc, prog_bar=True, logger=True, on_epoch=True)
        wandb.log({"validation/table": self.validation_table})

    def test_epoch_end(self, outputs):
        class_roc_auc, acc, preci, recall, f1 = self.evaluate(outputs)
        self.log("test/acc", acc, prog_bar=True, logger=True)
        self.log("test/preci", preci, prog_bar=True, logger=True)
        self.log("test/recall", recall, prog_bar=True, logger=True)
        self.log("test/f1", f1, prog_bar=True, logger=True)
        self.log(f"test/roc_auc", class_roc_auc, prog_bar=True, logger=True, on_epoch=True)
        wandb.log({"test/table": self.test_table})

    def get_event_type(self, outputs):
        label_map_file = "./index_label_map.json"
        with open(label_map_file, 'r') as f:
            index_label_map = json.load(f)
        labels_indices = [output.nonzero(as_tuple=True) for output in outputs]
        predicted_event_type = []
        for label_index in labels_indices:
            if not label_index[0].numel():
                event_type = ["non_event"]
                predicted_event_type.append(event_type)
            else:
                event_type = []
                for indices in label_index[0].detach():
                    event_type.append(index_label_map[str(indices.item())])
                    predicted_event_type.append(event_type)
        return predicted_event_type

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-5)
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

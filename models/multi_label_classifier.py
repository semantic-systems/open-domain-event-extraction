import pytorch_lightning as pl
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup, BertTokenizer
import torch
import torch.nn as nn
import torchmetrics
import json


class MavenModel(pl.LightningModule):
    def __init__(self, n_classes: int, pretrained_model_name_or_path: str, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path, return_dict=True)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()
        self.auroc = torchmetrics.AUROC(task="multilabel", num_labels=169).to(self.device)
        self.accuracy = torchmetrics.classification.MultilabelAccuracy(num_labels=169).to(self.device)
        self.preci = torchmetrics.classification.MultilabelPrecision(num_labels=169).to(self.device)
        self.recall = torchmetrics.classification.MultilabelRecall(num_labels=169).to(self.device)
        self.f1 = torchmetrics.classification.MultilabelF1Score(num_labels=169).to(self.device)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        sentences, labels = batch
        features = self.tokenizer.batch_encode_plus(sentences, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt', return_token_type_ids=False)
        loss, outputs = self.forward(features["input_ids"].to(device=self.device), features["attention_mask"].to(device=self.device), labels.to(device=self.device))
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        sentences, labels = batch
        features = self.tokenizer.batch_encode_plus(sentences, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt', return_token_type_ids=False)
        loss, outputs = self.forward(features["input_ids"].to(device=self.device), features["attention_mask"].to(device=self.device), labels.to(device=self.device))
        self.log("val_loss", loss, prog_bar=True, logger=True)
        prediction_int = torch.as_tensor((outputs - 0.5) > 0, dtype=torch.int32)
        data = {"sentences": sentences, "predictions": self.get_event_type(outputs), "labels": labels}
        self.log_table(key="validation", columns=list(data.keys()), data=data)
        return {"loss": loss, "predictions": outputs, "labels": labels, "prediction_int": prediction_int}

    def test_step(self, batch, batch_idx):
        sentences, labels = batch
        features = self.tokenizer.batch_encode_plus(sentences, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt', return_token_type_ids=False)
        loss, outputs = self.forward(features["input_ids"].to(device=self.device), features["attention_mask"].to(device=self.device), labels.to(device=self.device))
        self.log("test_loss", loss, prog_bar=True, logger=True)
        prediction_int = torch.as_tensor((outputs - 0.5) > 0, dtype=torch.int32)
        data = {"sentences": sentences, "predictions": self.get_event_type(outputs), "labels": labels}
        self.log_table(key="test", columns=list(data.keys()), data=data)
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
        self.log("valid/acc", acc, prog_bar=True, logger=True)
        self.log("valid/preci", preci, prog_bar=True, logger=True)
        self.log("valid/recall", recall, prog_bar=True, logger=True)
        self.log("valid/f1", f1, prog_bar=True, logger=True)
        self.log(f"train/roc_auc", class_roc_auc, prog_bar=True, logger=True, on_epoch=True)
        self.get_event_type(outputs)

    def test_epoch_end(self, outputs):
        class_roc_auc, acc, preci, recall, f1 = self.evaluate(outputs)
        self.log("test/acc", acc, prog_bar=True, logger=True)
        self.log("test/preci", preci, prog_bar=True, logger=True)
        self.log("test/recall", recall, prog_bar=True, logger=True)
        self.log("test/f1", f1, prog_bar=True, logger=True)
        self.log(f"train/roc_auc", class_roc_auc, prog_bar=True, logger=True, on_epoch=True)
        self.get_event_type(outputs)

    def get_event_type(self, outputs):
        label_map_file = "./index_label_map.json"
        with open(label_map_file, 'r') as f:
            index_label_map = json.load(f)
        labels_indices = [(output["prediction_int"] == 1).nonzero(as_tuple=False) for output in outputs]
        predicted_event_type = []
        for label_index in labels_indices:
            if not label_index.numel():
                event_type = ["non_event"]
                predicted_event_type.append(event_type)
            else:
                for indices in label_index.detach():
                    event_type = []
                    for i in indices:
                        # print(f"i {i}")
                        event_type.append(index_label_map[str(i.item())])
                    predicted_event_type.append(event_type)
        print("predicted_event_type", predicted_event_type)
        return predicted_event_type

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-4)
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

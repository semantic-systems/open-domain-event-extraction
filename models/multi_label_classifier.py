import pytorch_lightning as pl
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup, BertTokenizer
import torch
import torch.nn as nn
import torchmetrics


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
        self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=169).to(self.device)
        self.preci = torchmetrics.Precision(task="multilabel", average='macro', num_labels=169).to(self.device)
        self.recall = torchmetrics.Recall(task="multilabel", average='macro', num_labels=169).to(self.device)
        self.f1 = torchmetrics.F1Score(task="multilabel", average='macro', num_labels=169).to(self.device)

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
        loss, outputs  = self.forward(features["input_ids"].to(device=self.device), features["attention_mask"].to(device=self.device), labels.to(device=self.device))
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        sentences, labels = batch
        features = self.tokenizer.batch_encode_plus(sentences, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt', return_token_type_ids=False)
        loss, outputs = self.forward(features["input_ids"].to(device=self.device), features["attention_mask"].to(device=self.device), labels.to(device=self.device))
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def evaluate(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
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
        self.log("acc", acc, prog_bar=True, logger=True)
        self.log("preci", preci, prog_bar=True, logger=True)
        self.log("recall", recall, prog_bar=True, logger=True)
        self.log("f1", f1, prog_bar=True, logger=True)
        self.logger.experiment.add_scalar(f"roc_auc/Train", class_roc_auc, self.current_epoch)

    # def validation_epoch_end(self, outputs):
    #     class_roc_auc, acc, preci, recall, f1 = self.evaluate(outputs)
    #     self.log("acc", acc, prog_bar=True, logger=True)
    #     self.log("preci", preci, prog_bar=True, logger=True)
    #     self.log("recall", recall, prog_bar=True, logger=True)
    #     self.log("f1", f1, prog_bar=True, logger=True)
    #     self.logger.experiment.add_scalar(f"roc_auc/Train", class_roc_auc, self.current_epoch)
    #
    # def test_epoch_end(self, outputs):
    #     class_roc_auc, acc, preci, recall, f1 = self.evaluate(outputs)
    #     self.log("acc", acc, prog_bar=True, logger=True)
    #     self.log("preci", preci, prog_bar=True, logger=True)
    #     self.log("recall", recall, prog_bar=True, logger=True)
    #     self.log("f1", f1, prog_bar=True, logger=True)
    #     self.logger.experiment.add_scalar(f"roc_auc/Train", class_roc_auc, self.current_epoch)

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

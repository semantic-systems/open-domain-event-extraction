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
        self.auroc = torchmetrics.AUROC(task="multilabel", num_labels=169)

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
        features = self.tokenizer.batch_encode_plus(sentences, pad_to_max_length=True, truncation=True, return_attention_mask=True, return_tensors='pt', return_token_type_ids=False)
        loss, outputs = self(features["input_ids"], features["attention_mask"], labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        sentences, labels = batch
        features = self.tokenizer.batch_encode_plus(sentences, pad_to_max_length=True, truncation=True, return_attention_mask=True, return_tensors='pt', return_token_type_ids=False)
        loss, outputs = self(features["input_ids"], features["attention_mask"], labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        sentences, labels = batch
        features = self.tokenizer.batch_encode_plus(sentences, pad_to_max_length=True, truncation=True, return_attention_mask=True, return_tensors='pt', return_token_type_ids=False)
        loss, outputs = self(features["input_ids"], features["attention_mask"], labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        # for i, name in enumerate(range(169)):
        class_roc_auc = self.auroc(predictions, labels)
        self.logger.experiment.add_scalar(f"roc_auc/Train", class_roc_auc, self.current_epoch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

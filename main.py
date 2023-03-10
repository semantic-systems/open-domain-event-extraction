import pytorch_lightning as pl
import torch
from data_loader import MavenDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from models.multi_label_classifier import MavenModel

if __name__ == "__main__":
    torch.cuda.empty_cache()
    seed = 42
    BERT_MODEL_NAME = 'bert-base-cased'
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(seed)

    data_module = MavenDataModule(BERT_MODEL_NAME)
    bert_model = MavenModel(n_classes=169, pretrained_model_name_or_path=BERT_MODEL_NAME, n_training_steps=100,
                            n_warmup_steps=20)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    logger = TensorBoardLogger("lightning_logs", name="maven")
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=100,
        accelerator='gpu',
        devices=[1],
        fast_dev_run=False
    )
    trainer.fit(bert_model, datamodule=data_module)
    test_result = trainer.test(datamodule=data_module)
    torch.cuda.empty_cache()

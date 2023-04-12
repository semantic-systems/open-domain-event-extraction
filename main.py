import os
import pytorch_lightning as pl
import torch
import wandb
from data_loader import MavenDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from models.multi_label_classifier import MavenModel, InstructorModel, SentenceTransformersModel

if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed = 42
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(seed)
    wandb.login()

    data_module = MavenDataModule()
    model = SentenceTransformersModel(n_classes=169)
    # model = InstructorModel(n_classes=169)
    # model = MavenModel(n_classes=169, pretrained_model_name_or_path='roberta-base', n_training_steps=100, n_warmup_steps=20)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="miniLM-normalized-scl-best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    logger = WandbLogger(project="maven", name="miniLM/SCL-normalized")
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=200,
        # callbacks=[early_stopping_callback],
        callbacks=[early_stopping_callback, checkpoint_callback],
        accelerator='gpu',
        devices=[0],
        fast_dev_run=False
    )
    trainer.fit(model, datamodule=data_module)
    test_result = trainer.test(datamodule=data_module, ckpt_path='best')
    wandb.finish()
    torch.cuda.empty_cache()

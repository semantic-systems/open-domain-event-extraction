import os, yaml
import pytorch_lightning as pl
import torch
import wandb
from data_loader import MavenDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from models.multi_label_classifier import MavenModel, InstructorModel, SentenceTransformersModel


def main():
    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed = 42
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(seed)
    wandb.init()
    lr = 1e-04 #wandb.config.lr
    temperature = 0.1 #wandb.config.temperature
    alpha = 0.9 #wandb.config.alpha

    data_module = MavenDataModule()

    model = SentenceTransformersModel(n_classes=169, lr=lr, temperature=temperature, alpha=alpha)
    # model = InstructorModel(n_classes=169, lr=lr, temperature=temperature, alpha=alpha)
    # model = MavenModel(n_classes=169, lr=lr, temperature=temperature, alpha=alpha)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="miniLM-scl-best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="validation/f1",
        mode="max"
    )
    logger = WandbLogger(project="maven", name="miniLM/SCL/large-alpha")
    early_stopping_callback = EarlyStopping(monitor='validation/f1', patience=5, mode="max")

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=100,
        # callbacks=[early_stopping_callback],
        callbacks=[early_stopping_callback, checkpoint_callback],
        accelerator='gpu',
        devices=[0],
        fast_dev_run=False
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path='best')
    torch.cuda.empty_cache()
    print("Jobs done.")


def main_sweep():
    with open("./configs/miniLM/sweep_configuration.yaml", "r") as f:
        sweep_configuration = yaml.safe_load(f)
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='maven')
    wandb.agent(sweep_id, function=main, count=1)
    wandb.finish()


if __name__ == "__main__":
    main()


import os, yaml
import pytorch_lightning as pl
import torch
import wandb
from data_loader import MavenDataModule, TweetEvalDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from models.multi_label_classifier import MavenModel, InstructorModel
from models.single_label_classifier import SentenceTransformersModel


def main():
    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed = 42
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(seed)

    wandb.init()
    lr = wandb.config.lr
    temperature = wandb.config.temperature
    alpha = 1 # wandb.config.alpha
    num_augmentation = wandb.config.num_augmentation
    min_cluster_size = 50#wandb.config.min_cluster_size
    eps = 0.3
    min_samples = 10

    data_module = TweetEvalDataModule(batch_size=128)

    model = SentenceTransformersModel(n_classes=4, lr=lr, temperature=temperature, alpha=alpha,
                                      num_augmentation=num_augmentation, min_cluster_size=min_cluster_size,
                                      eps=eps, min_samples=min_samples)
    # model = InstructorModel(n_classes=169, lr=lr, temperature=temperature, alpha=alpha)
    # model = MavenModel(n_classes=169, lr=lr, temperature=temperature, alpha=alpha)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="emotion-best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="validation/f1",
        mode="min"
    )
    logger = WandbLogger(project="single-label/emotion/normalized", name="miniLM/normalized/sweep")
    early_stopping_callback = EarlyStopping(monitor='valid/contrastive loss', patience=15, mode="min", min_delta=0)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=200,
        # callbacks=[early_stopping_callback],
        callbacks=[early_stopping_callback, checkpoint_callback],
        accelerator='gpu',
        devices=[0, 1],
        fast_dev_run=False
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path='best')
    torch.cuda.empty_cache()
    print("Jobs done.")


def main_sweep():
    with open("configs/miniLM/emotion.yaml", "r") as f:
        sweep_configuration = yaml.safe_load(f)
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='when sb meets gmms')
    wandb.agent(sweep_id, function=main, count=36)
    wandb.finish()


if __name__ == "__main__":
    main_sweep()
    # main()

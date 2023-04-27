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
    lr = 0.00001 # wandb.config.lr
    temperature = wandb.config.temperature
    alpha = 1 # wandb.config.alpha
    num_augmentation = wandb.config.num_augmentation
    min_cluster_size = wandb.config.min_cluster_size
    eps = 0.3
    min_samples = 10

    data_module = TweetEvalDataModule(batch_size=512)

    model = SentenceTransformersModel(n_classes=20, lr=lr, temperature=temperature, alpha=alpha,
                                      num_augmentation=num_augmentation, min_cluster_size=min_cluster_size,
                                      eps=eps, min_samples=min_samples)
    # model = InstructorModel(n_classes=169, lr=lr, temperature=temperature, alpha=alpha)
    # model = MavenModel(n_classes=169, lr=lr, temperature=temperature, alpha=alpha)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="emoji-best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="valid/contrastive loss",
        mode="min"
    )
    logger = WandbLogger(project="single-label", name="miniLM/normalized/sweep")
    early_stopping_callback = EarlyStopping(monitor='valid/contrastive loss', patience=10, mode="min", min_delta=0)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=150,
        # callbacks=[early_stopping_callback],
        callbacks=[early_stopping_callback, checkpoint_callback],
        accelerator='gpu',
        devices=[1],
        fast_dev_run=False
    )
    trainer.fit(model, datamodule=data_module)
    # trainer.test(datamodule=data_module, ckpt_path='best')
    torch.cuda.empty_cache()
    print("Jobs done.")


def main_sweep():
    with open("configs/miniLM/emoji.yaml", "r") as f:
        sweep_configuration = yaml.safe_load(f)
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='single-label')
    wandb.agent(sweep_id, function=main, count=72)
    wandb.finish()


if __name__ == "__main__":
    main_sweep()
    # main()

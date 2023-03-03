from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
import torch
import pandas as pd
import ast
from typing import Optional
from transformers import BertTokenizer


class MavenMultiLabelClassificationDataset(Dataset):
    # This loads the data and converts it, make data rdy
    def __init__(self, path: str = "../../data/MAVEN/train.csv"):
        self.df = pd.read_csv(path)
        self.df_labels = self.df[['event_type']].fillna('["non_event"]').apply(self.deduplicate, axis=1)

        # convert to torch dtypes
        sentences = self.df[['sentence']].values
        self.dataset = sentences
        self.labels = self.multihot_encoder(self.df_labels)

    @staticmethod
    def deduplicate(df):
        return list(set(ast.literal_eval(df['event_type'])))

    @staticmethod
    def multihot_encoder(labels, dtype=torch.float32):
        """ Convert list of label lists into a 2-D multihot Tensor """
        label_set = set()
        for label_list in labels:
            label_set = label_set.union(set(label_list))
        label_set = sorted(label_set)

        multihot_vectors = []
        for label_list in labels:
            multihot_vectors.append([1 if x in label_list else 0 for x in label_set])

        # To keep track of which columns are which, set dtype to None and...
        if dtype is None:
            return pd.DataFrame(multihot_vectors, columns=label_set)
        return torch.Tensor(multihot_vectors).to(dtype)

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        sentence = self.dataset[idx][0]
        label = self.labels[idx]
        return sentence, label


class MavenDataModule(pl.LightningDataModule):

    def __init__(self, pretrained_model_name_or_path):
        super(MavenDataModule).__init__()
        self.prepare_data_per_node = True
        self._log_hyperparams = True
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally.
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: Optional[str] = None):
        self.train = MavenMultiLabelClassificationDataset("../../data/MAVEN/train.csv")
        self.test = MavenMultiLabelClassificationDataset("../../data/MAVEN/valid.csv")
        train_set_size = int(len(self.train) * 0.8)
        valid_set_size = len(self.train) - train_set_size
        self.train, self.validate = random_split(self.train, [train_set_size, valid_set_size])
        self.train, _ = random_split(self.train, [0.1, 0.9])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32, num_workers=8)

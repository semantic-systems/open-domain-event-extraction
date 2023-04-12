from pathlib import Path
import json
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
import torch
import pandas as pd
import ast
from typing import Optional


class MavenMultiLabelClassificationDataset(Dataset):
    # This loads the data and converts it, make data rdy
    def __init__(self, path: str = "../../data/MAVEN/train.csv"):
        self.df = pd.read_csv(str(Path(DATA_PATH, "train.csv").absolute()))
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
        label_map = {i: label for i, label in enumerate(label_set)}

        multihot_vectors = []
        for label_list in labels:
            multihot_vectors.append([1 if x in label_list else 0 for x in label_set])

        # To keep track of which columns are which, set dtype to None and...
        if dtype is None:
            return pd.DataFrame(multihot_vectors, columns=label_set)

        label_map_file = "./index_label_map.json"
        if not Path(label_map_file).exists():
            with open(label_map_file, 'w') as f:
                json.dump(label_map, f)

        return torch.Tensor(multihot_vectors).to(dtype)

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        sentence = self.dataset[idx][0]
        label = self.labels[idx]
        return sentence, label


DATA_PATH = "/export/home/huang/Projects/data/MAVEN/" # "../../data/MAVEN/"


class MavenDataModule(pl.LightningDataModule):

    def __init__(self):
        super(MavenDataModule).__init__()
        self.prepare_data_per_node = True
        self._log_hyperparams = True

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally.
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: Optional[str] = None):
        self.train = MavenMultiLabelClassificationDataset(str(Path(DATA_PATH, "train.csv").absolute()))
        self.test = MavenMultiLabelClassificationDataset(str(Path(DATA_PATH, "valid.csv").absolute()))
        train_set_size = int(len(self.train) * 0.8)
        valid_set_size = len(self.train) - train_set_size
        self.train, self.validate = random_split(self.train, [train_set_size, valid_set_size])
        # self.train, _ = random_split(self.train, [0.05, 0.95])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=64, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64, num_workers=8)

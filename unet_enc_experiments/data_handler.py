import copy
import os

import joblib
import nltk
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings

from global_configs import root_dir

warnings.filterwarnings('ignore')

class DiffusionDataset(Dataset):
    def __init__(self, df, transform, CFG):
        self.original_df = copy.deepcopy(df)
        self.stage = 4
        self.update_stage()

        self.transform = transform
        self.CFG = CFG
        self.root_dir = root_dir
        self.vocab = joblib.load(os.path.join(self.root_dir, "vocab.joblib"))


    def update_stage(self):
        self.upper_limit = int(len(self.original_df)/self.stage)
        self.df = self.original_df[0:self.upper_limit]
        if self.stage > 1:
            self.stage -= 1

    def use_entire_dataset(self):
        self.stage = 1
        self.update_stage()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.load(os.path.join(self.root_dir, row['filepath'][1:]))
        prompt = row['prompt']
        images = torch.Tensor(image)
        timesteps = 0
        with open(row['caption']) as f:
            caption = f.readlines()
        classes = np.zeros(len(self.vocab), dtype=np.float32)
        words = nltk.word_tokenize(prompt)
        for word in words:
            if word in self.vocab:
                classes[self.vocab[word]] = 1.
        embedding = np.load(os.path.join(self.root_dir, "sentence_embeddings", row['filepath'].split("/")[-1].split(".")[0]+"npy.npy"))
        return images, embedding, timesteps, caption[0], classes, row['filepath']

def get_dataloaders(
        trn_df,
        val_df,
        CFG
):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trn_dataset = DiffusionDataset(trn_df, transform, CFG)
    val_dataset = DiffusionDataset(val_df, transform, CFG)

    dataloaders = {}
    dataloaders['train'] = DataLoader(
        dataset=trn_dataset,
        shuffle=True,
        batch_size=CFG.batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False,
    )
    dataloaders['val'] = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=CFG.batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )
    return dataloaders


import copy
import os.path
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import nltk
from global_configs import root_dir

class VitDataset(Dataset):
    def __init__(self, images, transform,root_dir=root_dir,
                 root_train_dir=root_dir):
        self.original_df = copy.deepcopy(images)
        self.stage = 3
        self.update_stage()
        self.transform = transform
        self.root_dir = root_dir
        self.vocab = joblib.load(os.path.join(root_train_dir, "vocab.joblib"))

    def __len__(self):
        return len(self.df)

    def update_stage(self):
        self.upper_limit = int(len(self.original_df)/self.stage)
        self.df = self.original_df[0:self.upper_limit]
        if self.stage > 1:
            self.stage -= 1

    def use_entire_dataset(self):
        self.stage = 1
        self.update_stage()

    def __getitem__(self, idx):
        if idx < len(self.df):
            row = self.df.iloc[idx]
            file_name = row['filepath'].split("/")[-1]
            if "700k" in self.root_dir:
                for part in range(1, 9):
                    path = os.path.join(self.root_dir, f"images_part{part}", "images", file_name)
                    if os.path.exists(path):
                        final_path = path
                        break
                embedding = np.load(
                    os.path.join(self.root_dir, "sentence_embeddings", file_name.split(".")[0] + "npy.npy")).squeeze()
            else:
                for part in range(0,30):
                    path = os.path.join(self.root_dir, f"v{part}", file_name)
                    if os.path.exists(path):
                        final_path = path
                        break
                embedding = np.load(
                    os.path.join(self.root_dir, "sentence_embeddings", file_name.split(".")[0] + ".npy")).squeeze()

        image = Image.open(final_path)
        image = self.transform(image)
        classes = np.zeros(len(self.vocab), dtype=np.float32)
        words = nltk.word_tokenize(row['prompt'])
        for word in words:
            if word in self.vocab:
                classes[self.vocab[word]] = 1.

        return image, embedding, classes, row['filepath']

def get_dataloaders(trn_df,CFG):
    trn_df, val_df = train_test_split(trn_df, test_size=0.1, random_state=CFG.seed)

    transform = transforms.Compose([
        transforms.Resize(CFG.input_size, transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    trn_dataset = VitDataset(trn_df, transform)
    trn_dataloader = DataLoader(
        dataset=trn_dataset,
        shuffle=True,
        batch_size=CFG.batch_size,
        num_workers=2,
        drop_last=False
    )

    val_dataset = VitDataset(val_df, transform, root_dir=root_dir)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=CFG.batch_size,
        num_workers=2,
        drop_last=False
    )

    return trn_dataloader, val_dataloader


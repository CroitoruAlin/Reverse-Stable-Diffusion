import copy
import os.path

import joblib
import nltk
import numpy as np
import torch
from torch.utils.data import Dataset


class ClipEmbds(Dataset):
    def __init__(self, root_folder, df) -> None:
        super().__init__()
        self.root_folder = root_folder
        self.original_df = copy.deepcopy(df)
        self.stage = 2
        self.update_stage()
        self.vocab = joblib.load(os.path.join(root_folder, "vocab.joblib"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> tuple:
        row = self.df.iloc[index]

        image_name = row['filepath'].split("/")[-1].split(".")[0]

        clip_embd = np.load(os.path.join(self.root_folder, "clip_embeds_huge", image_name+".npy"))
        target = np.load(os.path.join(self.root_folder, "sentence_embeddings", image_name+"npy.npy"))

        classes = np.zeros(len(self.vocab), dtype=np.float32)
        words = nltk.word_tokenize(row['prompt'])
        for word in words:
            if word in self.vocab:
                classes[self.vocab[word]] = 1.

        return clip_embd, target, classes, row['filepath']

    def update_stage(self):
        self.upper_limit = int(len(self.original_df)/self.stage)
        self.df = self.original_df[0:self.upper_limit]
        if self.stage > 1:
            self.stage -= 1

    def use_entire_dataset(self):
        self.stage = 1
        self.update_stage()
        self.df = self.df[:100]
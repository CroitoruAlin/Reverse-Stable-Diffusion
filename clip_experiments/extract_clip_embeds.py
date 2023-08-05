import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from clip_experiments.model import ClipMLP
from ensemble_experiments.clip_modified import transform
from global_configs import root_dir


class DiffusionTestDataset(Dataset):
    def __init__(self, images):
        self.images = images
        self.preprocess = pickle.load(open("./trained_models/preprocess.pkl", 'rb'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.preprocess(image)
        return image, self.images[idx].stem


root_dataset = root_dir
images = list(Path(root_dataset).glob(f'*/images/*.png'))
dataset = DiffusionTestDataset(images)
dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=128,
        pin_memory=True,
        num_workers=4,
        drop_last=False
    )
device="cuda"
model = ClipMLP()
model.to(device)
dict_output = {}
model = torch.compile(model)
for X, image_name in tqdm(dataloader, leave=False):
    X = X.to(device).float()

    with torch.no_grad():
        X_out = model.encode_image(X)
        X_out = X_out.cpu().numpy()
        for i, features in enumerate(X_out):
            dict_output[image_name[i]] = features

for image_name in tqdm(dict_output):
    np.save(os.path.join(root_dataset, "clip_embeds_huge", image_name+".npy"), dict_output[image_name])




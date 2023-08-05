import os
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
root_dataset = "<TEST_DATA_SET>"
class DiffusionDataset(Dataset):

    def __init__(self, images, root_dataset, features_type):
        self.images = images
        self.root_dataset = root_dataset
        self.features_type = features_type

        self.dict_features_type = {}
        for feature_type in features_type:
            self.dict_features_type[feature_type] = joblib.load(os.path.join(root_dataset,"features", f"features_{feature_type}.joblib"))

    def __getitem__(self, idx):
        row = self.images[idx]
        file_name = row.stem
        features = []
        for type in self.features_type:
            features.append(self.dict_features_type[type][file_name])
        return np.vstack(features), file_name

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":

    images = list(Path(root_dataset).glob(f'*/*.png'))
    features_type = ["clip_mlp_images", "clip_mlp_images_2", "vit_gpt2", "unet", "vit_images", "swin_images", "knn"]
    weights = np.array([1.5, 1.2, 0.4, 0.4, 0.2, 1., 0.6])
    dataset = DiffusionDataset(images, root_dataset, features_type)
    val_dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )
    dict_features = {}
    for X, image_name in tqdm(val_dataloader, leave=False):
        X = F.normalize(X, dim=2)
        X = X.float().numpy()
        with torch.no_grad():
            X_out = np.average(X, axis=1, weights=weights)
            dict_features[image_name[0]]= X_out[0]
    joblib.dump(dict_features, os.path.join(root_dataset, "ensemble_predictions.joblib"))
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from transductive_kernel.extract_ensemble_preds import root_dataset


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)

    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity


class DiffusionDataset(Dataset):

    def __init__(self, images, root_dataset, features_type):
        self.images = images
        self.root_dataset = root_dataset
        self.features_type = features_type

        self.dict_features_type = {}
        for feature_type in features_type:
            self.dict_features_type[feature_type] = joblib.load(
                os.path.join(root_dataset, "features", f"features_{feature_type}.joblib"))

    def __getitem__(self, idx):
        row = self.images[idx]
        file_name = row.stem
        features = []
        for type in self.features_type:
            features.append(self.dict_features_type[type][file_name])
        return np.vstack(features)

    def __len__(self):
        return len(self.images)


class CFG:
    model_name = 'mlp_ensemble'
    batch_size = 128
    seed = 42
    features_type = ["clip_mlp_images", "blip", "unet", "vit_images", "swin_images", "knn", "krr"]
    weights = np.array([1.5, 1.2, 0.4, 0.4, 0.2, 1., 0.6, 0.3])


def predict(images, CFG):
    val_dataset = DiffusionDataset(images, CFG.root_dataset, CFG.features_type)

    val_dataloader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=CFG.batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preds = []
    for X in tqdm(val_dataloader, leave=False):
        X = F.normalize(X, dim=2)
        X = X.float().numpy()
        with torch.no_grad():
            X_out = np.average(X, axis=1, weights=CFG.weights)
        preds.extend(X_out)

    return np.array(preds)


images = list(Path(CFG.root_dataset).glob(f'*/*.png'))
initial_pred = predict(images, CFG)

st_model = SentenceTransformer('../all-MiniLM-L6-v2')

prompts = pd.read_csv(f"{root_dataset}/prompts.csv")
# #
prompts = prompts.to_dict()

id_idx = {v.split("/")[-1].split(".")[0]: k for k, v in prompts['filepath'].items()}

similarities_normal = []
for i, id in enumerate(tqdm(images)):
    idx = id_idx[id.stem]
    embedding = np.load(os.path.join(CFG.root_dataset, "sentence_embeddings", id.stem + ".npy"))
    prompt = prompts['prompt'][idx]
    if prompt is np.nan:
        continue
    # embedding = st_model.encode([prompt])
    similarities_normal.append(cosine_similarity(embedding, initial_pred[i]))

print("Weighted avg")
print(similarities_normal)
print(np.mean(similarities_normal))
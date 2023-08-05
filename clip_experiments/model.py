# import clip
import timm
import torch
from torch import nn
from transformers import CLIPModel, CLIPPreTrainedModel, CLIPConfig

from ensemble_experiments.clip_modified import CLIP
import torch.nn.functional as F

from global_configs import root_dir


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=1024, out_features=8192),
            nn.GELU(),
            nn.BatchNorm1d(num_features=8192),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=8192, out_features=8192),
            nn.GELU(),
            nn.BatchNorm1d(num_features=8192),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=8192, out_features=384)
        )
        self.classifier_head = nn.Linear(in_features=384, out_features=1000)

    def forward(self, input):
        x = self.network(input)
        preds = self.classifier_head(x)
        return F.normalize(x, p=2, dim=1), preds

class ClipMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.clip = CLIPModel.from_pretrained(f"../trained_models/clip-vit-h-14")
        self.mlp = MLP()

    def encode_image(self, x):
        return self.clip.get_image_features(x)

    def forward(self, x):
        x = self.clip.get_image_features(x)
        x = self.mlp(x)

        return x
    def use_mlp(self, x):
        x, preds = self.mlp(x)
        return x, preds
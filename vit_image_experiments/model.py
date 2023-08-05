import timm
from torch import nn
import torch.nn.functional as F

class VitModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.vit_model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=384
        )
        self.classifier_head = nn.Linear(384, 1000)

    def forward(self, input):
        x = self.vit_model(input)
        preds = self.classifier_head(x)

        return F.normalize(x, p=2, dim=1), preds
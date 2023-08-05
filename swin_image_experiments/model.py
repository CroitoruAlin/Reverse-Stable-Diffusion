import timm
import torch
from torch import nn
import torch.nn.functional as F

class SwinModel(nn.Module):

    def __init__(self, use_multi_label_output = False):
        super().__init__()
        self.vit_model = timm.create_model(
            'swinv2_large_window12to16_192to256_22kft1k',
            # "vit_base_patch16_224",
            pretrained=True,
            num_classes=384
        )
        # self.use_multi_label_output = use_multi_label_output
        self.classifier_head = nn.Linear(384, 1000)
        # if use_multi_label_output:
        #     self.embedding_head = nn.Linear(2536, 384)
        # else:
        # self.embedding_head = nn.Linear(768, 384)
        self.vit_model.set_grad_checkpointing()

    def forward(self, input):
        x = self.vit_model(input)
        # x = self.vit_model.forward_head(x, pre_logits=True)

        preds = self.classifier_head(x)
        # if self.use_multi_label_output:
        #     x = torch.concat((preds, x), dim=1)
        #
        # embedding = self.embedding_head(x)
        # return F.normalize(embedding, p=2, dim=1)
        return F.normalize(x, p=2, dim=1), preds

        # return F.normalize(x, p=2, dim=1)


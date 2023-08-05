import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss, CosineEmbeddingLoss, HuberLoss, CrossEntropyLoss


class HuberCosLoss(_Loss):

    def __init__(self, w=1.):
        super().__init__()
        self.cos_loss = CosineEmbeddingLoss()
        self.huber_loss = HuberLoss()
        self.w=w

    def forward(self, input1, input2, target):
        loss1 = self.cos_loss(input1,input2, target)
        loss2 = self.huber_loss(input1, input2)
        # return loss1, loss1, loss2
        return loss1+self.w*loss2, loss1, loss2

class ClipLoss(_Loss):

    def __init__(self, device="cuda", w=1.):
        super().__init__()
        self.ce = CrossEntropyLoss()
        self.cos_loss = CosineEmbeddingLoss()
        self.huber_loss = HuberLoss()
        self.w=w
        self.temperature = nn.Parameter(torch.Tensor([0.07])).to(device)
        self.device = device

    def forward(self, input1, input2, target):
        huber_loss = self.huber_loss(input1, input2)
        labels = torch.arange(0, input1.shape[0]).to(self.device)
        similarities = torch.matmul(input1, torch.transpose(input2, dim0=-1, dim1=-2)) * torch.exp(self.temperature)
        clip_loss = self.ce(similarities, labels)
        cos_loss = self.cos_loss(input1,input2,target)
        return cos_loss + clip_loss, clip_loss, huber_loss


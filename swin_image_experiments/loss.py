from torch.nn.modules.loss import _Loss, CosineEmbeddingLoss, BCEWithLogitsLoss, HuberLoss


class BCECossLoss(_Loss):

    def __init__(self, w=1.):
        super().__init__()
        self.cos_loss = CosineEmbeddingLoss()
        self.ce_loss = BCEWithLogitsLoss()
        self.huber_loss = HuberLoss()
        self.w=w

    def forward(self, emb_out, emb_target, target,preds=None, classes=None):
        loss1 = self.cos_loss(emb_out, emb_target, target)
        loss2 = self.ce_loss(preds, classes)
        loss3 = self.huber_loss(emb_out, emb_target)

        # return loss1+self.w*( loss3)
        return loss1+self.w*loss2 + loss3

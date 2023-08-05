from torch import nn

from stablediffusion.ldm.modules.diffusionmodules.encoder_unet import EncoderUNetModel


class UnetModel(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.unet = EncoderUNetModel(**configs)
        self.classification_head = nn.Linear(384, 1000)

    def forward(self, input, timesteps, context):
        output = self.unet(input,  timesteps=timesteps, context=context)
        labels = self.classification_head(output)
        return output, labels


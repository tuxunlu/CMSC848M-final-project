import torch
import torch.nn as nn

from .utils.vqvae import VQVAE
from .utils.transformer import Transformer


class LGBaseline(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs
        ):
        super(self, LGBaseline).__init__()
        self.vqvae = VQVAE(**kwargs)
        self.transformer = Transformer(**kwargs)
    
    def forward(self, image, caption):
        image_codebook = self.vqvae.encoder_forward(image)
        translated_language_codebook = self.transformer(caption)

        return image_codebook, translated_language_codebook
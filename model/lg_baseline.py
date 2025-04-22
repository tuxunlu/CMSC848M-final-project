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
        if 'pretrained_vqvae_path' not in kwargs:
            raise KeyError("Key pretrained_vqvae_path is not found in kwargs!")
        self.vqvae = self.load_vqvae_model(kwargs)
        self.transformer = Transformer(**kwargs)

        # Freeze parameters of VQVAE for inference only
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def load_vqvae_model(self, kwargs):
        model = VQVAE(**kwargs)
        path = kwargs['pretrained_vqvae_path']
        model.load_state_dict(path)
        return model
    
    def forward(self, image, caption):
        image_codebook = self.vqvae.encoder_forward(image)
        translated_language_codebook = self.transformer(caption)

        return image_codebook, translated_language_codebook
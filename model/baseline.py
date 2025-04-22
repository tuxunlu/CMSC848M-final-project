import torch
import torch.nn as nn

from .utils.vqvae import VQVAE
from .utils.transformer import Transformer


class Baseline(nn.Module):
    def __init__(
            self,
            # Transformer parameters
            src_vocab_size, 
            tgt_vocab_size, 
            d_model=512, 
            num_heads=8,
            num_encoder_layers=6, 
            num_decoder_layers=6, 
            dim_ff=1024, 
            dropout=0.1, 
            max_len=512,
            # VQVAE parameters
            h_dim=128, 
            res_h_dim=128, 
            n_res_layers=4,
            n_embeddings=1024, 
            embedding_dim=128, 
            beta=0.25, 
            save_img_embedding_map=False,
            pretrained_vqvae_path=None,
        ):
        super(Baseline, self).__init__()
        if 'pretrained_vqvae_path' is None:
            raise KeyError("Key pretrained_vqvae_path is None!")
        self.vqvae = self.load_vqvae_model(
            pretrained_vqvae_path, 
            h_dim=h_dim, 
            res_h_dim=res_h_dim, 
            n_res_layers=n_res_layers,
            n_embeddings=n_embeddings, 
            embedding_dim=embedding_dim, 
            beta=beta, 
            save_img_embedding_map=save_img_embedding_map
        )
        self.transformer = Transformer(src_vocab_size, 
            tgt_vocab_size, 
            d_model=d_model, 
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_ff=dim_ff, 
            dropout=dropout, 
            max_len=max_len
        )

        # Freeze parameters of VQVAE for inference only
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def load_vqvae_model(self, pretrained_vqvae_path, **kwargs):
        model = VQVAE(**kwargs)
        model.load_state_dict(torch.load(pretrained_vqvae_path))
        return model
    
    def forward(self, image, caption, src_mask):
        image_sentence = self.vqvae.encoder_forward(image)
        pred_image_sentence = self.transformer(caption, image_sentence, src_mask=src_mask)

        return image_sentence, pred_image_sentence
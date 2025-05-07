
import torch
import torch.nn as nn
import numpy as np
from .utils.encoder import Encoder
from .utils.quantizer import VectorQuantizer
from .utils.decoder import Decoder


class Vqvae(nn.Module):
    def __init__(self, h_dim=128, res_h_dim=128, n_res_layers=4,
                 n_embeddings=1024, embedding_dim=128, beta=0.25, save_img_embedding_map=False):
        super(Vqvae, self).__init__()
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, min_encoding_indices = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        B, _, H, W = z_e.shape  # expected (B, C, 30, 40) ⇒ 1200 tokens
        token_seq = min_encoding_indices.view(B, -1).long()  # (B, 1200)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, token_seq

    def encoder_forward(self, x):
        """
        Encode input image into a full discrete token sequence (B, 1200) of type torch.long
        """
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        _, _, _, _, min_encoding_indices = self.vector_quantization(z_e)  # shape: (B * H * W, 1)

        B, _, H, W = z_e.shape  # expected (B, C, 30, 40) ⇒ 1200 tokens
        token_seq = min_encoding_indices.view(B, -1).long()  # (B, 1200)

        return token_seq  # shape: (B, 1200), dtype=torch.long

    def decoder_forward(self, token_indices):
        """
        Decode from (B, 1200) token sequence of type torch.long to image
        Returns reconstructed image of shape (B, 3, H, W)
        """
        B, L = token_indices.shape  # L = 1200

        n_embeddings = self.vector_quantization.n_e
        embedding_dim = self.vector_quantization.e_dim
        device = token_indices.device

        # One-hot encoding of tokens
        one_hot = torch.zeros(B, L, n_embeddings, device=device)
        one_hot.scatter_(2, token_indices.unsqueeze(-1), 1)

        # Project one-hot to quantized embedding space: (B, L, embedding_dim)
        z_q = torch.matmul(one_hot, self.vector_quantization.embedding.weight)  # (B, L, emb_dim)

        # Reshape to image-shaped feature map: (B, emb_dim, 30, 40)
        z_q = z_q.view(B, 30, 40, embedding_dim).permute(0, 3, 1, 2).contiguous()

        x_hat = self.decoder(z_q)

        return x_hat  # (B, 3, H, W)


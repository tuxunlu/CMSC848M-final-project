
import torch
import torch.nn as nn
import numpy as np
from .encoder import Encoder
from .quantizer import VectorQuantizer
from .decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, h_dim=128, res_h_dim=128, n_res_layers=4,
                 n_embeddings=1024, embedding_dim=128, beta=0.25, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
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

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, min_encoding_indices

    def encoder_forward(self, x):
        """Encode input image into discrete token indices"""
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        _, _, _, _, min_encoding_indices = self.vector_quantization(z_e)
        return min_encoding_indices

    def decoder_forward(self, token_indices, shape):
        """
        Decode discrete tokens back to image space.
        Args:
            token_indices: (B * H * W, 1) token index tensor
            shape: original feature map shape (B, C, H, W) for reshaping
        Returns:
            everything in forward(): embedding_loss, x_hat, perplexity, z_q, min_encoding_indices
        """
        B, C, H, W = shape
        # Reconstruct one-hot encoding from token indices
        n_embeddings = self.vector_quantization.n_e
        embedding_dim = self.vector_quantization.e_dim
        device = token_indices.device

        one_hot = torch.zeros(token_indices.shape[0], n_embeddings, device=device)
        one_hot.scatter_(1, token_indices, 1)

        # Project one-hot to embedding space
        z_q = torch.matmul(one_hot, self.vector_quantization.embedding.weight)
        z_q = z_q.view(B, H, W, embedding_dim).permute(0, 3, 1, 2).contiguous()

        x_hat = self.decoder(z_q)

        return x_hat

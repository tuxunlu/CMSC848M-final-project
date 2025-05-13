import os
import re
import glob
import torch
import torch.nn as nn
import numpy as np
from .utils.encoder import Encoder
from .utils.quantizer import VectorQuantizer
from .utils.decoder import Decoder
from .model_interface_baseline import ModelInterfaceBaseline
from torchvision.utils import save_image
import matplotlib.pyplot as plt



class Vqvae(nn.Module):
    def __init__(self, h_dim=128, res_h_dim=128, n_res_layers=4,
                 n_embeddings=1024, embedding_dim=128, beta=0.25, save_img_embedding_map=False, downsample_height=4, downsample_width=4):
        super(Vqvae, self).__init__()
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim, downsample_height, downsample_width)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim, downsample_height, downsample_width)

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

        # Save results
        img_dir = "/fs/nexus-scratch/tuxunlu/git/CMSC848M-final-project/inference/imgs"
        os.makedirs(img_dir, exist_ok=True)
        files = glob.glob(os.path.join(img_dir, "*.png"))
        pattern = re.compile(r".*?_(\d+)\.png$|^(\d+)\.png$")

        indices = []
        for f in files:
            name = os.path.basename(f)
            m = pattern.match(name)
            if m:
                # one of the groups will have the number
                num = m.group(1) or m.group(2)
                indices.append(int(num))

        if indices:
            index = max(indices) + 1
        else:
            index = 0

        # now you can save, e.g.:
        # real_image_{index}.png, fake_image_{index}.png, etc.
        real_path = os.path.join(img_dir, f"real_image_{index}.png")
        gen_path = os.path.join(img_dir, f"gen_image_{index}.png")
        image_sentence_path = os.path.join(img_dir, f"image_sentence_{index}.png")
            
        batch_id = 1

        real_image = ModelInterfaceBaseline.denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        save_image(real_image[batch_id].float().div(255.0), real_path)
        gen_image = ModelInterfaceBaseline.denormalize(x_hat, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        save_image(gen_image[batch_id].float().div(255.0), gen_path)

        print("gen_image", gen_image)
        print("real_image", real_image)
        print("x_hat", x_hat)
        print("x", x)
        exit(0)

        idxs = token_seq[batch_id][:].cpu().numpy()
        grid = idxs.reshape(32, 32)
        print("grid.shape=", grid.shape)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.savefig(image_sentence_path)

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

    def decoder_forward(self, token_indices, downsample_height, downsample_width):
        """
        Decode from (B, 1200) token sequence of type torch.long to image
        Returns reconstructed image of shape (B, 3, H, W)
        """
        B, L = token_indices.shape

        n_embeddings = self.vector_quantization.n_e
        embedding_dim = self.vector_quantization.e_dim
        device = token_indices.device

        # One-hot encoding of tokens
        one_hot = torch.zeros(B, L, n_embeddings, device=device)
        one_hot.scatter_(2, token_indices.unsqueeze(-1), 1)

        # Project one-hot to quantized embedding space: (B, L, embedding_dim)
        z_q = torch.matmul(one_hot, self.vector_quantization.embedding.weight)  # (B, L, emb_dim)

        # Reshape to image-shaped feature map: (B, emb_dim, 30, 40) if downsample_height = 4, downsample_width = 4
        fmap_height = int(120 / downsample_height)
        fmap_width = int(160 / downsample_width)
        z_q = z_q.view(B, fmap_height, fmap_width, embedding_dim).permute(0, 3, 1, 2).contiguous()

        x_hat = self.decoder(z_q)

        return x_hat  # (B, 3, H, W)


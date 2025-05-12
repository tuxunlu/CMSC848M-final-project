
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .residual import ResidualStack


class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, downsample_height, downsample_width, height=640, width=480):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2
        self.height = height
        self.width = width

        second_deconv_stride = (2 if downsample_width >= 2 else 1, 2 if downsample_height >=2 else 1)
        first_deconv_stride = (2 if downsample_width == 4 else 1, 2 if downsample_height == 4 else 1)
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1),  # 40x30 → 40x30
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),                   # (B, h_dim, 40, 30)
            
            nn.ConvTranspose2d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),   # 40x30 → 80x60
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),  # 80x60 → 160x120
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, h_dim // 4, kernel_size=4, stride=second_deconv_stride, padding=1),  # 160x120 → 320x240
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 4, 3, kernel_size=4, stride=first_deconv_stride, padding=1),  # 320x240 → 640x480
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)[:, :, :self.width, :self.height]


if __name__ == "__main__":
    # Simulate encoder output: (B, embedding_dim=128, 40, 30)
    x = torch.randn(3, 128, 40, 30)
    decoder = Decoder(128, 128, 3, 64, 2, 4)
    out = decoder(x)
    print("Decoder output shape:", out.shape)

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

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


if __name__ == "__main__":
    # Simulate encoder output: (B, embedding_dim=128, 40, 30)
    x = torch.randn(3, 128, 40, 30)
    decoder = Decoder(128, 128, 3, 64, 2, 4)
    out = decoder(x)
    print("Decoder output shape:", out.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .residual import ResidualStack


class Decoder(nn.Module):
    def __init__(self,
                 in_dim, h_dim,
                 n_res_layers, res_h_dim,
                 downsample_height, downsample_width):
        super().__init__()
        kernel_4 = 4          # convenience: “full” kernel
        stride_2 = 2

        # --- strides that undo the two variable‑stride convs in the encoder -------
        # NOTE: stride tuple = (stride_height, stride_width) for ConvTranspose2d
        second_conv_stride = (
            2 if downsample_height >= 2 else 1,
            2 if downsample_width  >= 2 else 1,
        )
        first_conv_stride = (
            2 if downsample_height == 4 else 1,
            2 if downsample_width  == 4 else 1,
        )

        # --------------------------------------------------------------------------
        # Every transposed‑conv that has stride‑2 uses kernel 4, padding 1,
        # **no output_padding** (it already doubles the size exactly).
        #
        # If a stride component is 1, a 4×4 kernel would add +1 pixel along that
        # axis.  To keep the size unchanged we switch _that_ axis to a 3‑tap
        # kernel (mirror of the encoder’s 3×3 stride‑1 conv).
        # --------------------------------------------------------------------------
        def k3_or_k4(axis_stride):
            return 3 if axis_stride == 1 else 4

        last_kernel_size = (
            k3_or_k4(first_conv_stride[0]),   # height  kernel
            k3_or_k4(first_conv_stride[1]),   # width   kernel
        )
        last_padding = (1, 1)                 # works for both 3 and 4‑tap kernels
        last_output_padding = (0, 0)          # never needed after kernel fix

        self.inverse_conv_stack = nn.Sequential(
            # (1) holds spatial size
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),

            # (2)‑(3) fixed ×2 up‑sampling
            nn.ConvTranspose2d(h_dim,          h_dim,      kernel_size=4,
                               stride=stride_2, padding=1),           # ×2
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim,          h_dim // 2, kernel_size=4,
                               stride=stride_2, padding=1),           # ×2
            nn.ReLU(),

            # (4) variable ×1/×2 up‑sampling
            nn.ConvTranspose2d(h_dim // 2,     h_dim // 4, kernel_size=4,
                               stride=second_conv_stride, padding=1,
                               output_padding=(0, 0)),
            nn.ReLU(),

            # (5) final variable stride with axis‑specific kernel
            nn.ConvTranspose2d(h_dim // 4, 3,
                               kernel_size=last_kernel_size,
                               stride=first_conv_stride,
                               padding=last_padding,
                               output_padding=last_output_padding),
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


if __name__ == "__main__":
    # Simulate encoder output: (B, embedding_dim=128, 40, 30)
    x = torch.randn(3, 128, 40, 30)
    decoder = Decoder(128, 128, 3, 64)
    out = decoder(x)
    print("Decoder output shape:", out.shape)

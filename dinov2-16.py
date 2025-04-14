import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import types
from typing import Tuple

# ------------------------
# Positional Encoding Fix
# ------------------------
def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
    def interpolate_pos_encoding(self, tokens: torch.Tensor, w: int, h: int) -> torch.Tensor:
        npatch = tokens.shape[1] - 1  # without CLS
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = tokens.shape[-1]

        w0 = 1 + (w - patch_size) // stride_hw[1]
        h0 = 1 + (h - patch_size) // stride_hw[0]
        assert (w0 * h0 == npatch), (
            f"Expected {npatch} tokens, got {h0}x{w0} = {h0 * w0}"
        )

        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False, recompute_scale_factor=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    return interpolate_pos_encoding

# ------------------------
# Patch Stride Modifier
# ------------------------
def patch_vit_stride(model: nn.Module, stride: int) -> nn.Module:
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    if stride == patch_size:
        return model

    stride = (stride, stride)
    model.patch_embed.proj.stride = stride
    model.interpolate_pos_encoding = types.MethodType(_fix_pos_enc(patch_size, stride), model)
    return model

# ------------------------
# Setup
# ------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
patch_size = 14
# Load model and modify stride
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model = patch_vit_stride(model, stride=patch_size)
model = model.to(device).half().eval()  # convert to float16 and eval mode

# Dummy input
img = torch.randn(1, 3, 448, 448).to(device).half()

with torch.no_grad():
    tokens = model.forward_features(img)  # [B, 1 + N, C]
    patch_tokens = tokens["x_norm_patchtokens"]          # [B, N, C]

grid_size = 448//patch_size
patch_grid = patch_tokens[0].reshape(grid_size, grid_size, -1)  # [H, W, C]

print("Patch tokens shape:", patch_tokens.shape)     # e.g. [1, 784, 768]
print("Patch grid shape :", patch_grid.shape)        # e.g. [28, 28, 768]



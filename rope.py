import torch
import math

def apply_rope(x, offset=0):
    B, H, T, D = x.shape
    half = D // 2

    freqs = 1.0 / (10000 ** (torch.arange(0, half, device=x.device) / half))
    # позиции идут с offset: при генерации с KV-кэшем это длина уже закэшированного
    t = torch.arange(offset, offset + T, device=x.device, dtype=freqs.dtype)

    angles = torch.einsum("i,j->ij", t, freqs)

    cos = torch.cos(angles)[None, None, :, :]
    sin = torch.sin(angles)[None, None, :, :]

    x1, x2 = x[..., :half], x[..., half:]

    x_rot = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

    return x_rot

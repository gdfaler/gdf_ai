import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import apply_rope


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class Attention(nn.Module):
    def __init__(self, dim, n_heads, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x, cache=None):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # сколько токенов уже в кэше = позиция первого нового токена
        pos = cache["k"].size(2) if cache is not None else 0

        q = apply_rope(q, pos)
        k = apply_rope(k, pos)

        # KV CACHE (LLaMA style)
        if cache is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        T2 = att.size(-1)
        att = att.masked_fill(self.mask[pos:pos + T, :T2] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        new_cache = {
            "k": k,
            "v": v
        }

        return self.proj(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 4)
        self.w2 = nn.Linear(dim, dim * 4)
        self.w3 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, dim, n_heads, block_size):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, block_size)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim)

    def forward(self, x, cache=None):
        y, cache = self.attn(self.norm1(x), cache)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x, cache


class LLaMAMini(nn.Module):
    def __init__(self, vocab_size, block_size, dim=512, n_heads=8, n_layers=6):
        super().__init__()

        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            Block(dim, n_heads, block_size)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx, cache=None):
        B, T = idx.shape

        # cache: None или список {"k", "v"} — по одному на каждый блок
        pos = cache[0]["k"].size(2) if cache is not None else 0
        if pos + T > self.block_size:
            raise ValueError(
                f"контекст {pos + T} токенов больше block_size={self.block_size}"
            )

        x = self.tok_emb(idx)

        new_cache = []

        for i, block in enumerate(self.blocks):
            x, layer_cache = block(x, cache[i] if cache is not None else None)
            new_cache.append(layer_cache)

        x = self.norm(x)
        return self.head(x), new_cache
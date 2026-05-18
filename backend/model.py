import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()

        self.heads = heads
        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()

        att = att.masked_fill(~mask, torch.finfo(att.dtype).min)

        att = F.softmax(att, dim=-1)

        out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads=heads)

        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class LLaMAMini(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        dim=256,
        layers=4,
        heads=4
    ):
        super().__init__()

        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.blocks = nn.ModuleList(
            Block(dim, heads=heads) for _ in range(layers)
        )
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx):
        idx = idx[:, -self.block_size:]

        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return self.lm_head(x)
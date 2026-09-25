import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # считаем в float32: в fp16 квадраты легко переполняются
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x = (x.float() * torch.rsqrt(norm + self.eps)).type_as(x)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """RoPE: позиция кодируется поворотом пар координат q и k."""

    def __init__(self, head_dim, max_len, base=10000):
        super().__init__()
        half = head_dim // 2
        freqs = 1.0 / (base ** (torch.arange(half).float() / half))
        angles = torch.outer(torch.arange(max_len).float(), freqs)

        # cos/sin считаются один раз; в чекпоинт не попадают
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, pos):
        # x: (B, heads, T, head_dim), pos — позиция первого токена в x
        T = x.size(2)
        cos = self.cos[pos:pos + T]
        sin = self.sin[pos:pos + T]

        x1, x2 = x.float().chunk(2, dim=-1)

        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()

        self.heads = heads
        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, rope, cache=None):
        B, T, C = x.shape

        # сколько токенов уже в кэше = позиция первого нового токена
        pos = cache["k"].size(2) if cache is not None else 0

        q = self.q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        q = rope(q, pos)
        k = rope(k, pos)

        # KV-кэш: прошлые k/v не пересчитываются
        if cache is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)

        if cache is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            # один новый токен видит всё прошлое — маска не нужна
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            mask = torch.ones(T, k.size(2), dtype=torch.bool, device=x.device).tril(pos)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out), {"k": k, "v": v}


class FeedForward(nn.Module):
    """SwiGLU как в LLaMA: скрытый слой ~8/3·dim, без bias."""

    def __init__(self, dim):
        super().__init__()

        hidden = 64 * math.ceil(8 * dim / 3 / 64)

        self.w1 = nn.Linear(dim, hidden, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, dim, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads=heads)

        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x, rope, cache=None):
        y, cache = self.attn(self.norm1(x), rope, cache)
        x = x + y
        x = x + self.ff(self.norm2(x))
        return x, cache


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

        # сохраняется в чекпоинт, чтобы потом собрать ту же модель
        self.config = dict(
            vocab_size=vocab_size,
            block_size=block_size,
            dim=dim,
            layers=layers,
            heads=heads
        )

        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.rope = RotaryEmbedding(dim // heads, block_size)
        self.blocks = nn.ModuleList(
            Block(dim, heads=heads) for _ in range(layers)
        )
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

        # выходы блоков складываются в residual — уменьшаем их, чтобы
        # сигнал не рос с глубиной (как в GPT-2)
        for name, p in self.named_parameters():
            if name.endswith(("attn.out.weight", "ff.w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, cache=None):
        """idx: (B, T). cache: None или список {"k", "v"} по блокам из
        прошлого вызова. Возвращает (logits, новый cache)."""
        B, T = idx.shape

        pos = cache[0]["k"].size(2) if cache is not None else 0
        if pos + T > self.block_size:
            raise ValueError(
                f"контекст {pos + T} токенов больше block_size={self.block_size}"
            )

        x = self.token_emb(idx)

        new_cache = []

        for i, block in enumerate(self.blocks):
            x, layer_cache = block(
                x, self.rope, cache[i] if cache is not None else None
            )
            new_cache.append(layer_cache)

        x = self.norm(x)

        return self.lm_head(x), new_cache


def save_checkpoint(model, path, **extra):
    """extra — что-то сверх весов, например состояние оптимизатора."""
    torch.save({"config": model.config, "model": model.state_dict(), **extra}, path)


def read_checkpoint(path, device, vocab_size=None):
    """Читает чекпоинт и проверяет, что он подходит к коду и токенизатору."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"нет {path} — сначала обучи модель: python backend/train.py"
        )

    ckpt = torch.load(path, map_location=device)

    if "config" not in ckpt:
        raise RuntimeError(
            f"{path}: чекпоинт старого формата без настроек модели — "
            "переобучи: python backend/train.py"
        )

    if vocab_size is not None and ckpt["config"]["vocab_size"] != vocab_size:
        raise RuntimeError(
            f"словарь модели ({ckpt['config']['vocab_size']}) не совпадает "
            f"с токенизатором ({vocab_size}) — переобучи: python backend/train.py"
        )

    return ckpt


def load_checkpoint(path, device, vocab_size=None):
    """Собирает модель по настройкам из чекпоинта и строго грузит веса."""
    ckpt = read_checkpoint(path, device, vocab_size)

    model = LLaMAMini(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])

    return model

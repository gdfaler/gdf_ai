import os
import torch
import torch.nn.functional as F
import sentencepiece as spm

from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    BLOCK_SIZE,
    DIM,
    EPOCHS,
    HEADS,
    LAYERS,
    LR,
    VOCAB_SIZE,
)
from model import LLaMAMini
from dataset import TokenDataset
from generate_utils import build_prompt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.model")
TOKENS_PATH = os.path.join(BASE_DIR, "tokens.npy")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint.pt")


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


def get_vocab_size():
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    size = sp.get_piece_size()
    return min(size, VOCAB_SIZE)


def generate_sample(model, sp, prompt_text):
    model.eval()
    prompt = build_prompt(prompt_text)
    ids = sp.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(80):
            logits = model(x)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)

    out = sp.decode(x[0, len(ids):].tolist())
    return out.split("\n")[0].strip()


def train():
    vocab_size = get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    dataset = TokenDataset(TOKENS_PATH, BLOCK_SIZE)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = LLaMAMini(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        dim=DIM,
        layers=LAYERS,
        heads=HEADS
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    print("Training started")

    model.train()

    for epoch in range(EPOCHS):
        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if step % 20 == 0:
                print(
                    f"epoch {epoch} step {step} loss {loss.item():.4f}",
                    flush=True
                )

    torch.save(model.state_dict(), CHECKPOINT_PATH)

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    sample = generate_sample(model, sp, "Объясни attention")
    print("Sample:", sample.encode("utf-8", errors="replace").decode("utf-8"), flush=True)
    print("Training finished", flush=True)


if __name__ == "__main__":
    train()

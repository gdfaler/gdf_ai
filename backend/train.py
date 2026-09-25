import torch
import torch.nn.functional as F
import sentencepiece as spm

from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    BLOCK_SIZE,
    CHECKPOINT_PATH,
    DIM,
    EPOCHS,
    HEADS,
    LAYERS,
    LR,
    TOKENIZER_PATH,
    TOKENS_PATH,
)
from model import LLaMAMini, save_checkpoint
from dataset import TokenDataset
from generate_utils import build_prompt, clean_response, generate, should_stop


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


def generate_sample(model, sp, prompt_text):
    model.eval()
    ids = sp.encode(build_prompt(prompt_text))

    out = generate(
        model,
        ids,
        max_new_tokens=80,
        stop=lambda generated: should_stop(sp.decode(generated))
    )

    return clean_response(sp.decode(out))


def train():
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    vocab_size = sp.get_piece_size()
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
                logits, _ = model(x)

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

    save_checkpoint(model, CHECKPOINT_PATH)

    sample = generate_sample(model, sp, "Объясни attention")
    print("Sample:", sample.encode("utf-8", errors="replace").decode("utf-8"), flush=True)
    print("Training finished", flush=True)


if __name__ == "__main__":
    train()

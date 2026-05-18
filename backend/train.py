import os
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from model import LLaMAMini
from dataset import TokenDataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOKENS_PATH = os.path.join(BASE_DIR, "tokens.npy")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint.pt")


BATCH_SIZE = 16
BLOCK_SIZE = 256
EPOCHS = 1
LR = 3e-4


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


def train():
    dataset = TokenDataset(TOKENS_PATH, BLOCK_SIZE)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = LLaMAMini(
        vocab_size=4000,
        block_size=BLOCK_SIZE,
        dim=512,
        layers=8
    ).to(device)

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
            scaler.step(optimizer)
            scaler.update()

            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step} loss {loss.item():.4f}"
                )

            if step % 500 == 0:
                torch.save(
                    model.state_dict(),
                    CHECKPOINT_PATH
                )

    torch.save(model.state_dict(), CHECKPOINT_PATH)

    print("Training finished")
    train()
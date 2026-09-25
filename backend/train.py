import argparse
import csv
import math
import time

import torch
import torch.nn.functional as F
import sentencepiece as spm

from config import (
    BATCH_SIZE,
    BLOCK_SIZE,
    CHECKPOINT_BEST_PATH,
    CHECKPOINT_PATH,
    DIM,
    EVAL_INTERVAL,
    EVAL_ITERS,
    GRAD_CLIP,
    HEADS,
    LAYERS,
    LOG_INTERVAL,
    LR,
    MAX_STEPS,
    MIN_LR,
    TOKENIZER_PATH,
    TOKENS_PATH,
    TOKENS_PER_STEP,
    TRAIN_LOG_PATH,
    VAL_FRACTION,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from model import LLaMAMini, read_checkpoint, save_checkpoint
from dataset import get_batch, load_splits
from generate_utils import build_prompt, clean_response, generate, should_stop


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

torch.set_float32_matmul_precision("high")

# на эти вопросы модель отвечает при каждой проверке — видно, как она учится
SAMPLE_PROMPTS = ["Привет! Как дела?", "Объясни attention"]


def autocast():
    # bf16: в отличие от fp16 не переполняется, GradScaler не нужен
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))


def get_lr(step):
    """Линейный разгон WARMUP_STEPS шагов, затем косинусное затухание до MIN_LR."""
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS

    progress = min(1.0, (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS))
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def make_optimizer(model):
    # weight decay только для матриц; нормы (векторы) не штрафуем
    params = [p for p in model.parameters() if p.requires_grad]

    return torch.optim.AdamW(
        [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": WEIGHT_DECAY},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ],
        lr=LR,
        betas=(0.9, 0.95),
        fused=(device == "cuda")
    )


def compute_loss(model, x, y):
    with autocast():
        logits, _ = model(x)

        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )


@torch.no_grad()
def estimate_loss(model, data):
    model.eval()

    losses = [
        compute_loss(model, *get_batch(data, BATCH_SIZE, model.block_size, device)).item()
        for _ in range(EVAL_ITERS)
    ]

    model.train()
    return sum(losses) / len(losses)


def generate_sample(model, sp, prompt_text):
    model.eval()
    ids = sp.encode(build_prompt(prompt_text))

    out = generate(
        model,
        ids,
        max_new_tokens=80,
        stop=lambda generated: should_stop(sp.decode(generated))
    )

    model.train()
    return clean_response(sp.decode(out))


def train(resume=False):
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    vocab_size = sp.get_piece_size()
    print(f"Vocab size: {vocab_size}")

    if resume:
        ckpt = read_checkpoint(CHECKPOINT_PATH, device, vocab_size=vocab_size)

        if "optimizer" not in ckpt:
            raise RuntimeError(
                f"{CHECKPOINT_PATH}: нет состояния оптимизатора — начни обучение заново без --resume"
            )

        model = LLaMAMini(**ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model"])

        optimizer = make_optimizer(model)
        optimizer.load_state_dict(ckpt["optimizer"])

        start_step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]

        print(f"Resumed from step {start_step}")
    else:
        model = LLaMAMini(
            vocab_size=vocab_size,
            block_size=BLOCK_SIZE,
            dim=DIM,
            layers=LAYERS,
            heads=HEADS
        ).to(device)

        optimizer = make_optimizer(model)

        start_step = 0
        best_val_loss = float("inf")

    block_size = model.block_size

    train_data, val_data = load_splits(TOKENS_PATH, block_size, VAL_FRACTION)
    print(f"Tokens: train {len(train_data):,}, val {len(val_data):,}")

    accum_steps = max(1, TOKENS_PER_STEP // (BATCH_SIZE * block_size))
    tokens_per_step = accum_steps * BATCH_SIZE * block_size

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Tokens per step: {tokens_per_step:,} ({BATCH_SIZE} x {block_size} x {accum_steps} accum)")

    def save(step):
        save_checkpoint(
            model,
            CHECKPOINT_PATH,
            optimizer=optimizer.state_dict(),
            step=step,
            best_val_loss=best_val_loss
        )

    with open(TRAIN_LOG_PATH, "a" if resume else "w", newline="", encoding="utf-8") as log_file:
        log = csv.writer(log_file)

        if not resume:
            log.writerow(["step", "tokens", "lr", "train_loss", "val_loss", "tokens_per_sec"])

        print("Training started")

        model.train()

        step = start_step
        loss_sum = torch.zeros((), device=device)
        loss_count = 0
        timer_steps = 0
        timer = time.perf_counter()

        try:
            while step < MAX_STEPS:
                lr = get_lr(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                optimizer.zero_grad(set_to_none=True)

                for _ in range(accum_steps):
                    x, y = get_batch(train_data, BATCH_SIZE, block_size, device)
                    loss = compute_loss(model, x, y) / accum_steps
                    loss.backward()
                    loss_sum += loss.detach()

                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                step += 1
                loss_count += 1
                timer_steps += 1

                do_eval = step % EVAL_INTERVAL == 0 or step == MAX_STEPS

                if step % LOG_INTERVAL != 0 and not do_eval:
                    continue

                train_loss = loss_sum.item() / loss_count
                tokens_per_sec = timer_steps * tokens_per_step / (time.perf_counter() - timer)

                val_loss = ""
                if do_eval:
                    val_loss = estimate_loss(model, val_data)

                line = (
                    f"step {step} loss {train_loss:.4f} lr {lr:.2e} "
                    f"{tokens_per_sec:,.0f} tok/s"
                )
                if do_eval:
                    line += f" | val {val_loss:.4f}"
                print(line, flush=True)

                log.writerow([
                    step,
                    step * tokens_per_step,
                    f"{lr:.3e}",
                    f"{train_loss:.4f}",
                    f"{val_loss:.4f}" if do_eval else "",
                    f"{tokens_per_sec:.0f}"
                ])
                log_file.flush()

                if do_eval:
                    for prompt in SAMPLE_PROMPTS:
                        print(f"  {prompt} → {generate_sample(model, sp, prompt)}", flush=True)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, CHECKPOINT_BEST_PATH)

                    save(step)

                loss_sum.zero_()
                loss_count = 0
                timer_steps = 0
                timer = time.perf_counter()

        except KeyboardInterrupt:
            save(step)
            print(f"\nStopped at step {step}, saved. Continue: python backend/train.py --resume")
            return

    sample = generate_sample(model, sp, "Объясни attention")
    print("Sample:", sample.encode("utf-8", errors="replace").decode("utf-8"), flush=True)
    print(f"Best val loss: {best_val_loss:.4f}")
    print("Training finished", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение LLaMAMini (параметры — в config.py)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="продолжить с checkpoint.pt (веса, оптимизатор и шаг)"
    )
    train(resume=parser.parse_args().resume)

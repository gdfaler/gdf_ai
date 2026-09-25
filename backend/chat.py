import torch
import sentencepiece as spm

from config import (
    CHECKPOINT_PATH,
    MAX_GEN_TOKENS,
    REPETITION_PENALTY,
    TOKENIZER_PATH,
    TOP_K,
)
from model import load_checkpoint
from generate_utils import build_prompt, clean_response, generate, should_stop


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))

model = load_checkpoint(CHECKPOINT_PATH, device, vocab_size=sp.get_piece_size())
model.eval()

print("Chat ready (exit — выход)")

while True:

    inp = input("\nYou: ")

    if inp == "exit":
        break

    out = generate(
        model,
        sp.encode(build_prompt(inp)),
        MAX_GEN_TOKENS,
        temperature=0.8,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        stop=lambda generated: should_stop(sp.decode(generated))
    )

    print("AI:", clean_response(sp.decode(out)))

from pathlib import Path
import numpy as np
import sentencepiece as spm

BASE_DIR = Path(__file__).resolve().parent

TRAIN_TXT_PATH = BASE_DIR / "train.txt"
TOKENIZER_PATH = BASE_DIR / "tokenizer.model"
TOKENS_PATH = BASE_DIR / "tokens.npy"

print("Loading tokenizer:", TOKENIZER_PATH)

sp = spm.SentencePieceProcessor(
    model_file=str(TOKENIZER_PATH)
)

with open(TRAIN_TXT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

tokens = sp.encode(text)

tokens = np.array(tokens, dtype=np.uint16)

np.save(TOKENS_PATH, tokens)

print("Saved tokens:", TOKENS_PATH)
print("Total tokens:", len(tokens))
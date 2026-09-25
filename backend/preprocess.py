import numpy as np
import sentencepiece as spm

from config import TOKENIZER_PATH, TOKENS_PATH, TRAIN_TXT_PATH

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

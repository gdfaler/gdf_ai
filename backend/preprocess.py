import os
import json
import numpy as np
import sentencepiece as spm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.model")
DATASET_PATH = os.path.join(BASE_DIR, "dataset.jsonl")
TOKENS_PATH = os.path.join(BASE_DIR, "tokens.npy")


sp = spm.SentencePieceProcessor(
    model_file=TOKENIZER_PATH
)


tokens = []

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        q = item.get("question", "")
        a = item.get("answer", "")

        text = f"User: {q}\nAI: {a}"

        ids = sp.encode(text)

        tokens.extend(ids)


tokens = np.array(tokens, dtype=np.uint16)

np.save(TOKENS_PATH, tokens)

print("Saved:", TOKENS_PATH)
print("Tokens:", len(tokens))
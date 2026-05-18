import os
import json
import sentencepiece as spm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset.jsonl")
TRAIN_TXT_PATH = os.path.join(BASE_DIR, "train.txt")


with open(DATASET_PATH, "r", encoding="utf-8") as f, \
     open(TRAIN_TXT_PATH, "w", encoding="utf-8") as out:

    for line in f:
        item = json.loads(line)

        q = item.get("question", "")
        a = item.get("answer", "")

        text = f"User: {q}\nAI: {a}\n"

        out.write(text)


spm.SentencePieceTrainer.train(
    input=TRAIN_TXT_PATH,
    model_prefix=os.path.join(BASE_DIR, "tokenizer"),
    vocab_size=4000,
    model_type="bpe",
    character_coverage=1.0,
    normalization_rule_name="identity",
    max_sentence_length=100000
)

print("Tokenizer created")
from pathlib import Path
import sentencepiece as spm

BASE_DIR = Path(__file__).resolve().parent

TRAIN_TXT_PATH = BASE_DIR / "train.txt"
TOKENIZER_PREFIX = BASE_DIR / "tokenizer"

print("TRAIN FILE:", TRAIN_TXT_PATH)

spm.SentencePieceTrainer.train(
    input=str(TRAIN_TXT_PATH),
    model_prefix=str(TOKENIZER_PREFIX),
    vocab_size=512,
    model_type="bpe",
    normalization_rule_name="identity",
    character_coverage=1.0,
    max_sentence_length=100000,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

print("Tokenizer trained")
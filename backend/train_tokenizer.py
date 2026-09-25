import sentencepiece as spm

from config import TOKENIZER_PREFIX, TRAIN_TXT_PATH, VOCAB_SIZE

print("TRAIN FILE:", TRAIN_TXT_PATH)

spm.SentencePieceTrainer.train(
    input=str(TRAIN_TXT_PATH),
    model_prefix=str(TOKENIZER_PREFIX),
    vocab_size=VOCAB_SIZE,
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

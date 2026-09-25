import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="data.txt",
    model_prefix="tokenizer",
    vocab_size=32000,
    model_type="bpe",

    normalization_rule_name="identity",

    character_coverage=1.0,
    byte_fallback=True
)
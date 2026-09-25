from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Файлы пайплайна
TRAIN_TXT_PATH = BASE_DIR / "train.txt"
TOKENIZER_PREFIX = BASE_DIR / "tokenizer"
TOKENIZER_PATH = BASE_DIR / "tokenizer.model"
TOKENS_PATH = BASE_DIR / "tokens.npy"
CHECKPOINT_PATH = BASE_DIR / "checkpoint.pt"

# Модель (app.py и chat.py берут размеры из чекпоинта, если он есть)
BLOCK_SIZE = 128
DIM = 256
LAYERS = 4
HEADS = 4
VOCAB_SIZE = 512

# Обучение
BATCH_SIZE = 8
EPOCHS = 30
LR = 3e-4

# Генерация
MAX_GEN_TOKENS = 120
TOP_K = 40
REPETITION_PENALTY = 1.15

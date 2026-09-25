from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Файлы пайплайна
TRAIN_TXT_PATH = BASE_DIR / "train.txt"
TOKENIZER_PREFIX = BASE_DIR / "tokenizer"
TOKENIZER_PATH = BASE_DIR / "tokenizer.model"
TOKENS_PATH = BASE_DIR / "tokens.npy"
CHECKPOINT_PATH = BASE_DIR / "checkpoint.pt"            # последнее состояние (+ оптимизатор)
CHECKPOINT_BEST_PATH = BASE_DIR / "checkpoint_best.pt"  # веса с лучшим val loss
TRAIN_LOG_PATH = BASE_DIR / "train_log.csv"

# Модель (app.py и chat.py берут размеры из чекпоинта, если он есть)
BLOCK_SIZE = 128
DIM = 256
LAYERS = 4
HEADS = 4
VOCAB_SIZE = 512

# Обучение
BATCH_SIZE = 32          # окон за один прогон модели
TOKENS_PER_STEP = 4096   # токенов на шаг оптимизатора; больше BATCH_SIZE·BLOCK_SIZE → накопление градиентов
MAX_STEPS = 2000
WARMUP_STEPS = 100       # линейный разгон LR, дальше косинусное затухание до MIN_LR
LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# Контроль обучения
VAL_FRACTION = 0.02      # хвост tokens.npy на проверку (не меньше 4 окон)
EVAL_INTERVAL = 200      # val loss, примеры ответов и сохранение каждые N шагов
EVAL_ITERS = 20          # батчей на оценку val loss
LOG_INTERVAL = 50

# Генерация
MAX_GEN_TOKENS = 120
TOP_K = 40
REPETITION_PENALTY = 1.15

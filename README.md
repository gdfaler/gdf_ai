# GDF AI

Локальный чат-бот на PyTorch — мини-трансформер в духе LLaMA с веб-интерфейсом в стиле ChatGPT.

Обучается на вашем тексте, работает полностью offline, без внешних API.

---

## Возможности

- **Собственная модель** — архитектура LLaMAMini: RMSNorm, multi-head attention, SwiGLU FFN
- **Локальный inference** — генерация текста на CPU или GPU
- **SentencePiece** — BPE-токенизатор, обучаемый на вашем корпусе
- **Flask API** — REST-эндпоинт `/chat` для генерации
- **Современный UI** — тёмная тема, glassmorphism, анимация печати ответов

---

## Как это работает

```
train.txt  →  train_tokenizer.py  →  tokenizer.model
                    ↓
              preprocess.py  →  tokens.npy
                    ↓
               train.py  →  checkpoint.pt
                    ↓
                app.py  →  http://127.0.0.1:5000
```

1. **train_tokenizer.py** — обучает BPE-токенизатор на `train.txt`
2. **preprocess.py** — кодирует текст в массив токенов `tokens.npy`
3. **train.py** — обучает нейросеть и сохраняет веса в `checkpoint.pt`
4. **app.py** — запускает веб-сервер с чат-интерфейсом

---

## Быстрый старт

### 1. Установка

```bash
git clone <repo-url>
cd gdf_ai

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> Для GPU установите PyTorch с поддержкой CUDA: [pytorch.org](https://pytorch.org/get-started/locally/)

### 2. Подготовка данных

Отредактируйте `backend/train.txt` — это обучающий корпус.  
Текущий файл компактный (~12 KB), обучение занимает 1–3 минуты на GPU.

### 3. Обучение

```bash
python backend/train_tokenizer.py
python backend/preprocess.py
python backend/train.py
```

В консоли вы увидите loss на каждом 50-м шаге:

```
Device: cuda
Training started
epoch 0 step 0 loss 8.4610
epoch 0 step 50 loss 5.2341
...
Training finished
```

### 4. Запуск

```bash
python backend/app.py
```

Откройте в браузере: **http://127.0.0.1:5000**

---

## Структура проекта

```
gdf_ai/
├── backend/
│   ├── app.py              # Flask-сервер и inference
│   ├── model.py            # LLaMAMini (трансформер)
│   ├── train.py            # Скрипт обучения
│   ├── train_tokenizer.py  # Обучение токенизатора
│   ├── preprocess.py       # Текст → tokens.npy
│   ├── dataset.py          # PyTorch Dataset
│   ├── train.txt           # Обучающий корпус
│   ├── tokenizer.model     # SentencePiece (генерируется)
│   ├── tokens.npy          # Токены (генерируется)
│   └── checkpoint.pt       # Веса модели (генерируется)
├── frontend/
│   ├── templates/
│   │   └── index.html      # Страница чата
│   └── static/
│       ├── style.css       # Стили
│       └── app.js          # Логика чата
├── requirements.txt
└── README.md
```

---

## Параметры модели

| Параметр     | Значение | Описание                        |
|-------------|----------|---------------------------------|
| `vocab_size`| 4000     | Размер словаря токенизатора     |
| `block_size`| 256      | Длина контекста (окно)          |
| `dim`       | 512      | Размерность скрытого слоя       |
| `layers`    | 8        | Количество transformer-блоков   |
| `heads`     | 8        | Головы в multi-head attention   |

Параметры обучения в `train.py`:

| Параметр      | Значение | Описание              |
|--------------|----------|-----------------------|
| `BATCH_SIZE` | 16       | Размер батча          |
| `EPOCHS`     | 1        | Число эпох            |
| `LR`         | 3e-4     | Learning rate (AdamW) |

---

## API

### `POST /chat`

```json
// Request
{ "message": "Объясни attention" }

// Response
{ "response": "Attention — механизм..." }
```

---

## Советы

- **Больше данных → лучше ответы.** Замените `train.txt` на свой корпус и переобучите пайплайн целиком.
- **Быстрые эксперименты** — используйте компактный `train.txt` (как сейчас).
- **Нет GPU** — обучение работает на CPU, но медленнее.
- **Нет checkpoint.pt** — приложение запустится с необученной моделью и выведет предупреждение.

---

## Стек

| Компонент    | Технология     |
|-------------|----------------|
| Модель      | PyTorch        |
| Токенизатор | SentencePiece  |
| Backend     | Flask          |
| Frontend    | HTML / CSS / JS|

---

## Лицензия

MIT

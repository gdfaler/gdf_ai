# GDF AI

Локальный чат-бот на PyTorch — мини-трансформер в духе LLaMA с веб-интерфейсом в стиле ChatGPT.

Обучается на вашем тексте, работает полностью offline, без внешних API.

---

## Возможности

- **Собственная модель** — архитектура LLaMAMini: RMSNorm, RoPE, SwiGLU FFN, общие веса эмбеддингов и выходного слоя
- **Быстрая генерация** — KV-кэш: промпт прогоняется один раз, дальше модель получает по одному токену
- **SentencePiece** — BPE-токенизатор, обучаемый на вашем корпусе
- **Flask API** — REST-эндпоинт `/chat` для генерации
- **Современный UI** — тёмная тема, glassmorphism, анимация печати ответов
- **Консольный чат** — `backend/chat.py` для быстрых проверок без браузера

---

## Как это работает

```
train.txt  →  train_tokenizer.py  →  tokenizer.model
                    ↓
              preprocess.py  →  tokens.npy
                    ↓
               train.py  →  checkpoint.pt (веса + настройки модели)
                    ↓
       app.py  →  http://127.0.0.1:5000
       chat.py →  чат в консоли
```

1. **train_tokenizer.py** — обучает BPE-токенизатор на `train.txt`
2. **preprocess.py** — кодирует текст в массив токенов `tokens.npy`
3. **train.py** — обучает нейросеть и сохраняет `checkpoint.pt`
4. **app.py** — запускает веб-сервер с чат-интерфейсом; **chat.py** — чат в консоли

Все пути и параметры — в `backend/config.py`.

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

`requirements.txt` ставит PyTorch со сборкой под CUDA 12.8 (подходит для RTX 50xx). Без видеокарты модель работает на CPU, только медленнее.

### 2. Подготовка данных

Отредактируйте `backend/train.txt` — это обучающий корпус (~30 КБ диалогов в формате `Пользователь: … / GDF AI: …`).

Чтобы обучить модель на другом тексте, поменяйте `TRAIN_TXT_PATH` в `backend/config.py`. Большие корпуса кладите в `data/` — эта папка не попадает в git.

### 3. Обучение

```bash
python backend/train_tokenizer.py
python backend/preprocess.py
python backend/train.py
```

На текущем `train.txt` обучение занимает ~5 минут на RTX 5070 Ti. В консоли видно loss каждые 20 шагов и пример ответа в конце:

```
Device: cuda
Vocab size: 512
Parameters: 3,344,640
Training started
epoch 0 step 0 loss 6.3110
...
Sample: Attention — механизм, при котором модель решает, на какие части входа смотреть сильнее. ...
Training finished
```

`tokenizer.model`, `tokens.npy` и `checkpoint.pt` генерируются и в git не хранятся — после `git clone` пайплайн нужно прогнать один раз.

### 4. Запуск

```bash
python backend/app.py
```

Откройте в браузере: **http://127.0.0.1:5000**

Или чат в консоли (`exit` — выход):

```bash
python backend/chat.py
```

---

## Структура проекта

```
gdf_ai/
├── backend/
│   ├── app.py              # Flask-сервер
│   ├── chat.py             # Консольный чат
│   ├── config.py           # Пути и параметры
│   ├── model.py            # LLaMAMini + сохранение/загрузка чекпоинта
│   ├── generate_utils.py   # Генерация с KV-кэшем, промпт, семплирование
│   ├── train.py            # Скрипт обучения
│   ├── train_tokenizer.py  # Обучение токенизатора
│   ├── preprocess.py       # Текст → tokens.npy
│   ├── dataset.py          # PyTorch Dataset
│   ├── train.txt           # Обучающий корпус
│   ├── tokenizer.model     # SentencePiece (генерируется)
│   ├── tokens.npy          # Токены (генерируется)
│   └── checkpoint.pt       # Веса и настройки модели (генерируется)
├── frontend/
│   ├── templates/
│   │   └── index.html      # Страница чата
│   └── static/
│       ├── style.css       # Стили
│       └── app.js          # Логика чата
├── tests/
│   └── test_model.py       # Тесты модели и KV-кэша
├── data/                   # Большие корпуса (не в git)
├── archive/                # Старые веса (не в git)
├── requirements.txt
└── README.md
```

---

## Параметры модели

Задаются в `backend/config.py` и сохраняются в `checkpoint.pt`: `app.py` и `chat.py` собирают модель по настройкам из чекпоинта.

| Параметр     | Значение | Описание                        |
|-------------|----------|---------------------------------|
| `VOCAB_SIZE`| 512      | Размер словаря токенизатора     |
| `BLOCK_SIZE`| 128      | Длина контекста (окно)          |
| `DIM`       | 256      | Размерность скрытого слоя       |
| `LAYERS`    | 4        | Количество transformer-блоков   |
| `HEADS`     | 4        | Головы в multi-head attention   |

Параметры обучения:

| Параметр      | Значение | Описание              |
|--------------|----------|-----------------------|
| `BATCH_SIZE` | 8        | Размер батча          |
| `EPOCHS`     | 30       | Число эпох            |
| `LR`         | 3e-4     | Learning rate (AdamW) |

Если чекпоинт не подходит к коду или токенизатору (старый формат, другой размер словаря), `app.py` и `chat.py` остановятся с понятной ошибкой — переобучите модель.

---

## API

### `POST /chat`

```json
// Request
{ "message": "Объясни attention", "max_tokens": 120, "temperature": 0.0 }

// Response
{ "response": "Attention — механизм..." }
```

`max_tokens` и `temperature` необязательны.

---

## Тесты

```bash
python -m unittest discover -s tests -v
```

Проверяют, что генерация с KV-кэшем совпадает с полным прогоном, что длинный контекст не ломает генерацию и что чекпоинт сохраняется и загружается без потерь.

---

## Советы

- **Больше данных → лучше ответы.** Сейчас модель просто запоминает `train.txt`; чтобы она обобщала, нужен корпус в тысячи раз больше.
- **Нет GPU** — обучение работает на CPU, но медленнее.
- **Нет checkpoint.pt** — веб-сервер запустится с необученной моделью и выведет предупреждение.

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

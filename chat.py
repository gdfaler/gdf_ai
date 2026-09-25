import torch
import torch.nn.functional as F

from model import LLaMAMini
from tokenizer import BPETokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

# tokenizer
tok = BPETokenizer("tokenizer.model")

# model
model = LLaMAMini(
    vocab_size=tok.vocab_size,
    block_size=128
).to(device)

model.load_state_dict(
    torch.load("llama_mini.pt", map_location=device)
)

model.eval()


def sample(logits, temperature=0.8, top_k=40, top_p=0.9):
    logits = logits[:, -1, :] / temperature

    # TOP-K (topk уже отсортирован по убыванию)
    probs, idx = torch.topk(F.softmax(logits, dim=-1), top_k)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # TOP-P: оставляем минимальный набор, который набирает top_p
    # (первый токен остаётся всегда, так что сумма не бывает 0)
    cumulative = torch.cumsum(probs, dim=-1)
    probs[cumulative - probs > top_p] = 0
    probs = probs / probs.sum(dim=-1, keepdim=True)

    return idx.gather(-1, torch.multinomial(probs, 1))


def generate(ids, max_new_tokens=80):
    block_size = model.block_size

    ids = list(ids)
    generated = []

    # сначала прогоняем весь промпт (последние block_size токенов),
    # дальше подаём по одному новому токену, остальное берётся из кэша
    x = ids[-block_size:]
    cache = None
    cached = 0

    with torch.no_grad():

        for _ in range(max_new_tokens):

            logits, cache = model(
                torch.tensor([x], dtype=torch.long, device=device),
                cache
            )
            cached += len(x)

            token = sample(logits).item()

            generated.append(token)
            ids.append(token)

            # ответ AI — одна строка: стоп на переводе строки
            if "\n" in tok.decode(generated).lstrip():
                break

            if cached < block_size:
                x = [token]
            else:
                # окно заполнено: пересчитываем по последним block_size токенам
                x = ids[-block_size:]
                cache = None
                cached = 0

    return tok.decode(generated).strip().split("\n")[0]


print("Chat ready")

while True:

    inp = input("\nYou: ")

    if inp == "exit":
        break

    # тот же формат, что и диалоги в data.txt
    tokens = tok.encode(f"User: {inp}\nAI:")

    print("AI:", generate(tokens))
import torch
import torch.nn.functional as F


def build_prompt(user_text):
    return f"Пользователь: {user_text.strip()}\nGDF AI:"


def apply_repetition_penalty(logits, token_ids, penalty):
    if penalty <= 1.0 or not token_ids:
        return logits

    scores = logits.clone()
    for tid in set(token_ids):
        score = scores[tid]
        scores[tid] = score / penalty if score > 0 else score * penalty

    return scores


def sample_token(logits, temperature, top_k):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0 and top_k < logits.size(-1):
        values, _ = torch.topk(logits, top_k)
        cutoff = values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, torch.finfo(logits.dtype).min)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model, ids, max_new_tokens, temperature=0.0, top_k=0,
             repetition_penalty=1.0, stop=None):
    """Продолжает ids, возвращает список новых токенов.

    Промпт прогоняется один раз, дальше модель получает по одному токену,
    остальное берёт из KV-кэша. stop(generated_ids) -> True останавливает.
    """
    device = next(model.parameters()).device
    block_size = model.block_size

    ids = list(ids)
    generated = []

    x = ids[-block_size:]
    cache = None
    cached = 0

    for _ in range(max_new_tokens):
        logits, cache = model(
            torch.tensor([x], dtype=torch.long, device=device),
            cache
        )
        cached += len(x)

        next_logits = apply_repetition_penalty(
            logits[0, -1],
            generated,
            repetition_penalty
        )
        token = sample_token(next_logits.unsqueeze(0), temperature, top_k).item()

        generated.append(token)
        ids.append(token)

        if stop is not None and stop(generated):
            break

        if cached < block_size:
            x = [token]
        else:
            # окно заполнено: пересчитываем по последним block_size токенам
            x = ids[-block_size:]
            cache = None
            cached = 0

    return generated


def should_stop(decoded_text):
    if "Пользователь:" in decoded_text:
        return True

    if "===" in decoded_text:
        return True

    stripped = decoded_text.strip()
    if "\n" in decoded_text and len(stripped) > 8:
        return True

    if len(stripped) > 280:
        return True

    return False


def clean_response(text):
    for ch in ("⁇", "\u2047", "\ufffd"):
        text = text.replace(ch, "")

    for stop in ("Пользователь:", "===", "\n\n"):
        if stop in text:
            text = text.split(stop)[0]

    if "\n" in text:
        text = text.split("\n")[0]

    return text.strip()

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

import os
import torch
import sentencepiece as spm

from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

from config import (
    BLOCK_SIZE,
    DIM,
    HEADS,
    LAYERS,
    MAX_GEN_TOKENS,
    REPETITION_PENALTY,
    TOP_K,
    VOCAB_SIZE,
)
from model import LLaMAMini
from generate_utils import (
    apply_repetition_penalty,
    build_prompt,
    clean_response,
    sample_token,
    should_stop,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.model")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint.pt")


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

sp = spm.SentencePieceProcessor(
    model_file=TOKENIZER_PATH
)

vocab_size = min(sp.get_piece_size(), VOCAB_SIZE)

model = LLaMAMini(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    dim=DIM,
    layers=LAYERS,
    heads=HEADS
).to(device)

if os.path.exists(CHECKPOINT_PATH):
    try:
        state_dict = torch.load(
            CHECKPOINT_PATH,
            map_location=device
        )

        model.load_state_dict(
            state_dict,
            strict=False
        )
        print("Checkpoint loaded")
    except (OSError, RuntimeError) as e:
        print(f"Warning: failed to load checkpoint ({e}), using untrained model")
else:
    print("Warning: checkpoint not found, using untrained model")

model.eval()

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)


def generate_reply(user_text, max_tokens, temperature):
    prompt = build_prompt(user_text)
    ids = sp.encode(prompt)
    prompt_len = len(ids)

    x = torch.tensor([ids], dtype=torch.long).to(device)
    generated_ids = []

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x)
            next_logits = logits[:, -1, :].squeeze(0)

            next_logits = apply_repetition_penalty(
                next_logits,
                generated_ids,
                REPETITION_PENALTY
            )

            next_token = sample_token(
                next_logits.unsqueeze(0),
                temperature,
                TOP_K
            )

            token_id = next_token.item()
            generated_ids.append(token_id)
            x = torch.cat([x, next_token], dim=1)

            partial = sp.decode(generated_ids)
            if should_stop(partial):
                break

    out = sp.decode(generated_ids) if generated_ids else ""
    return clean_response(out)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}

    text = data.get("message", "")
    max_tokens = min(max(int(data.get("max_tokens", MAX_GEN_TOKENS)), 1), 512)
    temperature = max(float(data.get("temperature", 0.0)), 0.0)

    if not text.strip():
        return jsonify({"response": ""})

    out = generate_reply(text, max_tokens, temperature)

    return jsonify({
        "response": out
    })


if __name__ == "__main__":
    app.run(debug=True)

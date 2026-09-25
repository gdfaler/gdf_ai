import torch
import sentencepiece as spm

from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

from config import (
    BLOCK_SIZE,
    CHECKPOINT_PATH,
    DIM,
    HEADS,
    LAYERS,
    MAX_GEN_TOKENS,
    REPETITION_PENALTY,
    TOKENIZER_PATH,
    TOP_K,
)
from model import LLaMAMini, load_checkpoint
from generate_utils import (
    build_prompt,
    clean_response,
    generate,
    should_stop,
)


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

sp = spm.SentencePieceProcessor(
    model_file=str(TOKENIZER_PATH)
)

if CHECKPOINT_PATH.exists():
    # несовместимый чекпоинт — ошибка, а не тихий запуск на случайных весах
    model = load_checkpoint(CHECKPOINT_PATH, device, vocab_size=sp.get_piece_size())
    print("Checkpoint loaded")
else:
    print("Warning: checkpoint not found, using untrained model")

    model = LLaMAMini(
        vocab_size=sp.get_piece_size(),
        block_size=BLOCK_SIZE,
        dim=DIM,
        layers=LAYERS,
        heads=HEADS
    ).to(device)

model.eval()

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)


def generate_reply(user_text, max_tokens, temperature):
    ids = sp.encode(build_prompt(user_text))

    generated_ids = generate(
        model,
        ids,
        max_tokens,
        temperature=temperature,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        stop=lambda generated: should_stop(sp.decode(generated))
    )

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

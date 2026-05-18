import os
import torch
import sentencepiece as spm

from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

from model import LLaMAMini


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.model")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint.pt")


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

sp = spm.SentencePieceProcessor(
    model_file=TOKENIZER_PATH
)

vocab_size = sp.get_piece_size()

model = LLaMAMini(
    vocab_size=vocab_size,
    block_size=256,
    dim=512,
    layers=8
).to(device)

state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    state_dict,
    strict=False
)

model.eval()

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json

    text = data.get("message", "")

    ids = sp.encode(text)

    x = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(80):
            logits = model(x)

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True
            )

            x = torch.cat([x, next_token], dim=1)

    out = sp.decode(x[0].tolist())

    return jsonify({
        "response": out
    })
    app.run(debug=True)
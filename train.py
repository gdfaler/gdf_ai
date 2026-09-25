import torch
import torch.nn.functional as F
from model import LLaMAMini
from tokenizer import BPETokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = BPETokenizer("tokenizer.model")

text = open("data.txt", encoding="utf-8").read()
data = torch.tensor(tok.encode(text), dtype=torch.long)

block_size = 128
batch_size = 16

model = LLaMAMini(tok.vocab_size, block_size).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

def get_batch():
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

for step in range(3000):
    x, y = get_batch()

    logits, _ = model(x)

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 200 == 0:
        print(step, loss.item())

torch.save(model.state_dict(), "llama_mini.pt")
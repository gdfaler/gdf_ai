import torch

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)
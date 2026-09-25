import numpy as np
import torch


def load_splits(path, block_size, val_fraction):
    """Токены с диска → (train, val).

    Файл открывается через memmap: в память попадают только прочитанные окна,
    поэтому подходит и для корпусов в гигабайты. Проверочная часть — хвост
    файла, но не меньше 4 окон.
    """
    data = np.load(path, mmap_mode="r")

    n_val = max(int(len(data) * val_fraction), 4 * (block_size + 1))

    if len(data) - n_val < 2 * (block_size + 1):
        raise ValueError(
            f"в {path} всего {len(data)} токенов — мало для block_size={block_size}"
        )

    return data[:-n_val], data[-n_val:]


def get_batch(data, batch_size, block_size, device):
    """Случайные окна: x — вход, y — те же токены со сдвигом на один."""
    ix = torch.randint(len(data) - block_size, (batch_size,)).tolist()

    x = np.stack([data[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + block_size + 1] for i in ix]).astype(np.int64)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    if device == "cuda":
        return (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True)
        )

    return x.to(device), y.to(device)

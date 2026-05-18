import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, path, block_size):
        self.data = np.load(path)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = torch.tensor(
            self.data[idx:idx + self.block_size],
            dtype=torch.long
        )

        y = torch.tensor(
            self.data[idx + 1:idx + self.block_size + 1],
            dtype=torch.long
        )

        return x, y
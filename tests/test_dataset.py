import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from dataset import get_batch, load_splits  # noqa: E402


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tokens.npy"
        np.save(self.path, np.arange(1000, dtype=np.uint16))

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_is_tail_with_minimum_size(self):
        train, val = load_splits(self.path, block_size=16, val_fraction=0.02)
        # 2% от 1000 = 20 токенов меньше минимума 4 окна по 17
        self.assertEqual(len(val), 68)
        self.assertEqual(len(train) + len(val), 1000)
        self.assertEqual(int(val[0]), int(train[-1]) + 1)

    def test_too_few_tokens_raises(self):
        with self.assertRaises(ValueError):
            load_splits(self.path, block_size=200, val_fraction=0.02)

    def test_batch_targets_are_shifted_inputs(self):
        train, _ = load_splits(self.path, block_size=16, val_fraction=0.02)
        x, y = get_batch(train, batch_size=64, block_size=16, device="cpu")

        self.assertEqual(tuple(x.shape), (64, 16))
        self.assertTrue((y[:, :-1] == x[:, 1:]).all())
        self.assertTrue((y[:, -1] == x[:, -1] + 1).all())
        self.assertLess(int(y.max()), len(train))


if __name__ == "__main__":
    unittest.main()

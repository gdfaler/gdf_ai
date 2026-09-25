import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from model import LLaMAMini, load_checkpoint, save_checkpoint  # noqa: E402
from generate_utils import generate  # noqa: E402


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = LLaMAMini(
            vocab_size=100, block_size=32, dim=64, layers=2, heads=4
        ).eval()
        self.idx = torch.randint(0, 100, (2, 32))

    def full_logits(self):
        with torch.no_grad():
            return self.model(self.idx)[0]

    def test_cache_token_by_token_matches_full_forward(self):
        with torch.no_grad():
            out, cache = self.model(self.idx[:, :10])
            parts = [out]
            for t in range(10, 32):
                out, cache = self.model(self.idx[:, t:t + 1], cache)
                parts.append(out)

        torch.testing.assert_close(torch.cat(parts, 1), self.full_logits(), atol=1e-5, rtol=1e-4)

    def test_cache_chunk_matches_full_forward(self):
        with torch.no_grad():
            out1, cache = self.model(self.idx[:, :10])
            out2, _ = self.model(self.idx[:, 10:], cache)

        torch.testing.assert_close(torch.cat([out1, out2], 1), self.full_logits(), atol=1e-5, rtol=1e-4)

    def test_context_overflow_raises(self):
        with torch.no_grad():
            _, cache = self.model(self.idx)
            with self.assertRaises(ValueError):
                self.model(self.idx[:, :1], cache)

    def test_generate_past_block_size(self):
        out = generate(self.model, self.idx[0, :30].tolist(), max_new_tokens=50)
        self.assertEqual(len(out), 50)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(self.model, path)
            loaded = load_checkpoint(path, "cpu", vocab_size=100).eval()

            with self.assertRaises(RuntimeError):
                load_checkpoint(path, "cpu", vocab_size=512)

        with torch.no_grad():
            torch.testing.assert_close(loaded(self.idx)[0], self.full_logits())

    def test_old_checkpoint_format_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save(self.model.state_dict(), path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path, "cpu")


if __name__ == "__main__":
    unittest.main()

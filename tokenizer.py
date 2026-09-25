import sentencepiece as spm

class BPETokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)

    @property
    def vocab_size(self):
        return self.sp.vocab_size()
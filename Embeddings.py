import torch
import torch.nn as nn

class TokenEmbeddings(nn.Module):

    def __init__(self, vocab_size: int, emb_size: int) -> None:
        self.matrix = nn.Embedding(vocab_size, emb_size)
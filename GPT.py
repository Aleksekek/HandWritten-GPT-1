import torch
import torch.nn as nn
from Embeddings import TokenEmbeddings, PositionalEmbeddings
from Decoder import Decoder

class GPT(nn.Module):

    def __init__(
            self, 
            vocab_size: int, 
            max_seq_len: int,
            emb_size: int,
            num_heads: int,
            head_size: int,
            num_layers: int,
            dropout: float = 0.1,
            device: str = 'cpu'
        ) -> None:
        super().__init__()
        self.token_embeddings = TokenEmbeddings(vocab_size=vocab_size, emb_size=emb_size)
        self.positional_embeddings = PositionalEmbeddings(max_seq_len=max_seq_len, emb_size=emb_size)
        self.dropout = nn.Dropout(p=dropout)
        self.decoders = nn.Sequential(*[
            Decoder(
                num_heads=num_heads,
                emb_size=emb_size, 
                head_size=head_size,
                max_seq_len=max_seq_len,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(emb_size, vocab_size)


    def forward(self, x):
        tokens = self.token_embeddings(x)
        positions = self.positional_embeddings(x.shape[1])
        x = tokens + positions
        x = self.dropout(x)
        x = self.decoders(x)
        x = self.linear(x)
        return x
import torch
import torch.nn as nn
from Attention import HeadAttention, MultiHeadAttention
from FFN import FeedForward

class Decoder(nn.Module):
    
    def __init__(
        self,
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.miltihead_attention = MultiHeadAttention(
            num_heads=num_heads,
            emb_size=emb_size,
            head_size=head_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.ffn = FeedForward(emb_size=emb_size, dropout=dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.miltihead_attention(x)
        output = output + x
        output = self.norm1(output)
        x = self.ffn(output)
        x = x + output
        x = self.norm2(x)
        return x
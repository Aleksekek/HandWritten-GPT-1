import torch
import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, emb_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(emb_size, emb_size*4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(emb_size*4, emb_size)
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
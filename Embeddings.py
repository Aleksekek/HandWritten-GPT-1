import torch
import torch.nn as nn

class TokenEmbeddings(nn.Module):

    def __init__(self, vocab_size: int, emb_size: int) -> None:
        super().__init__()
        self.matrix = nn.Embedding(vocab_size, emb_size)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x имеет размерность: batch_size × seq_len
        # self.matrix(x) автоматически ищет вектора по индексам
        # возвращает: batch_size × seq_len × emb_size
        return self.matrix(x)
    

class PositionalEmbeddings(nn.Module):

    def __init__(self, max_seq_len: int, emb_size: int) -> None:
        super().__init__()
        self.matrix = nn.Embedding(max_seq_len, emb_size)


    def forward(self, seq_len: int) -> torch.Tensor:
        # Создаем тензор позиций от 0 до seq_len-1
        positions = torch.arange(seq_len)
        
        # Вытаскиваем вектора для этих позиций
        # Возвращаем тензор размером seq_len × emb_size
        return self.matrix(positions)
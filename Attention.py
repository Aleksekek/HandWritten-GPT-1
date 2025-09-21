import torch
import torch.nn as nn

class HeadAttention(nn.Module):

    def __init__(self, emb_size: int, head_size: int, max_seq_len: int) -> None:
        super().__init__()
        self.W_k = nn.Linear(emb_size, head_size)
        self.W_q = nn.Linear(emb_size, head_size)
        self.W_v = nn.Linear(emb_size, head_size)
        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len))

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x имеет размерность: batch_size × seq_len × emb_size

        K_matrix = self.W_k(x)  # batch_size × seq_len × head_size
        Q_matrix = self.W_q(x)  # batch_size × seq_len × head_size  
        V_matrix = self.W_v(x)  # batch_size × seq_len × head_size

        # Из batch_size × seq_len × head_size в batch_size × head_size × seq_len
        K_transposed = K_matrix.transpose(1, 2)

        # batch_size × seq_len × head_size @ batch_size × head_size × seq_len
        # Результат: batch_size × seq_len × seq_len
        attention_scores = torch.matmul(Q_matrix, K_transposed)
        attention_scores = attention_scores / (K_matrix.size(-1) ** 0.5)

        seq_len = x.size(1)
        masked_attention = attention_scores.masked_fill(
            self.mask[:seq_len, :seq_len] == 0, float('-inf')
        )
        attention_weights = masked_attention.softmax(dim=-1)

        # batch_size × seq_len × seq_len @ batch_size × seq_len × head_size
        # Результат: batch_size × seq_len × head_size
        output = torch.matmul(attention_weights, V_matrix)

        return output
    

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                HeadAttention(
                    emb_size=emb_size, head_size=head_size, max_seq_len=max_seq_len
                )
                for _ in range(num_heads)
            ]
        )
        self.linear = nn.Linear(head_size*num_heads, emb_size)
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, x: float):
        outputs = [head(x) for head in self.heads]
        output = torch.cat(outputs, dim=-1)
        output = self.linear(output)
        output = self.dropout(output)
        return output
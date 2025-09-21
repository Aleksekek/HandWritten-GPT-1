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
        self.max_seq_len = max_seq_len
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
    

    def generate(self, x: torch.Tensor, max_new_tokens: int, do_sample: bool) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = x[:, -self.max_seq_len:]

            logits = self.forward(context)
            
            next_token_probs = logits[:, -1, :].softmax(dim=-1)
            if not do_sample:
                next_token = next_token_probs.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(next_token_probs, 1)

            x = torch.cat([x, next_token], dim=-1)

        return x
    

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'emb_size': self.emb_size,
            'num_heads': self.num_heads,
            'head_size': self.head_size,
            'num_layers': self.num_layers
        }, path)

 
    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_seq_len=checkpoint['max_seq_len'],
            emb_size=checkpoint['emb_size'],
            num_heads=checkpoint['num_heads'],
            head_size=checkpoint['head_size'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model
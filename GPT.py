import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
        self.device = device
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
    

    def generate(
            self, 
            x: torch.Tensor, 
            max_new_tokens: int, 
            do_sample: bool, 
            temperature: float = 1.0, 
            top_k: int = None, top_p: float = None # type: ignore
        ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = x[:, -self.max_seq_len:]

            logits = self.forward(context) / temperature
            logits = logits[:, -1, :]

            if do_sample:
                # --- top_k ---
                if top_k is not None:
                    topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
                    mask = torch.zeros_like(logits, dtype=torch.uint8)
                    mask.scatter_(1, topk_idx, 1)
                    logits = logits.masked_fill(mask == 0, -float('Inf'))

                # --- top_p ---
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    probs = sorted_logits.softmax(dim=-1)
                    cum_probs = probs.cumsum(dim=-1)

                    mask = cum_probs < top_p
                    mask[..., 0] = 1

                    mask = mask.to(torch.uint8)
                    unsorted_mask = torch.zeros_like(mask)
                    unsorted_mask.scatter_(1, sorted_indices, mask)
                    logits = logits.masked_fill(unsorted_mask == 0, -float('Inf'))

            next_token_probs = logits.softmax(dim=-1)
            if not do_sample:
                next_token = next_token_probs.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(next_token_probs, 1)

            x = torch.cat([x, next_token], dim=-1)

        return x
    

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader, num_epochs: int, learning_rate: float):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        entropy_loss = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.train()
            for i_t in train_loader:
                inputs, targets = i_t
                x = self.forward(inputs)
                batch_size, seq_len, vocab_size = x.shape
                x = torch.reshape(x, [batch_size*seq_len, vocab_size])
                targets = torch.flatten(targets)
                loss = entropy_loss(x, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.eval()
            with torch.no_grad():
                for val in valid_loader:
                    inputs, targets = val
                    x = self.forward(inputs)
                    batch_size, seq_len, vocab_size = x.shape
                    x = torch.reshape(x, [batch_size*seq_len, vocab_size])
                    targets = torch.flatten(targets)
                    loss = entropy_loss(x, targets)
    

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
import torch
from torch.utils.data import Dataset
import torch.nn as nn

class GetData(Dataset):
    
    def __init__(self, data: list, seq_len: int, device: str) -> None:
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.device = device


    def __len__(self):
        return len(self.data) - self.seq_len - 1


    def __getitem__(self, idx: int):
        x = torch.tensor(self.data[idx:idx + self.seq_len])
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1])
        return (x.to(self.device), y.to(self.device))
import torch
from torch.utils.data import Dataset


class IMDBDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data[idx]["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(self.data[idx]["attention_mask"], dtype=torch.long)
        label = torch.tensor(self.data[idx]["label"], dtype=torch.long)
        return input_ids, attention_mask, label

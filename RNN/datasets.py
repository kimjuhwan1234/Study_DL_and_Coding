import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, submission=False):
        self.data = data
        self.window_size = 5 * 4
        self.sub = submission

    def __len__(self):
        if self.sub:
            return 1
        return len(self.data) - self.window_size

    def __getitem__(self, index):
        start_index = index
        end_index = index + self.window_size

        if end_index + 1 > len(self.data):
            raise IndexError("Index out of bounds. Reached the end of the dataset.")

        if self.sub:
            return torch.tensor(self.data.iloc[start_index:, :].values, dtype=torch.float32)

        X_train = self.data.iloc[start_index:end_index, :]
        y_train = self.data.iloc[end_index:end_index + 1, 0]

        X_train_tensor = torch.tensor(X_train.values)
        y_train_tensor = torch.tensor(y_train.values)

        return X_train_tensor, y_train_tensor.float()

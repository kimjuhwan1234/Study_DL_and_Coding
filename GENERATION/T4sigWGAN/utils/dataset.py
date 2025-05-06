import torch
import pandas as pd
from torch.utils.data import Dataset
from ..utils.utils import sample_indices


def train_test_split(
        x: torch.Tensor,
        train_test_ratio: float,
        device: str):
    size = x.shape[0]
    train_set_size = int(size * train_test_ratio)

    indices_train = sample_indices(size, train_set_size, device)
    indices_test = torch.LongTensor([i for i in range(size) if i not in indices_train])

    x_train = x[indices_train]
    x_test = x[indices_test]
    return x_train, x_test


def rolling_window(x: torch.Tensor, window_size: int):
    windowed_data = []
    for t in range(x.shape[0] - window_size + 1):
        window = x[t:t + window_size, :]
        windowed_data.append(window)
    return torch.stack(windowed_data, dim=0)


def pct_change(x: torch.Tensor):
    prev = x[:, :-1, :]
    curr = x[:, 1:, :]
    zero_mask = prev == 0
    pct = (curr - prev) / prev
    pct[zero_mask] = 0
    return pct


class StockTimeSeriesDataset(Dataset):
    def __init__(self, csv_path='data/sp500.csv', tickers=None, window_size=31):
        super().__init__()
        if tickers is None:
            tickers = ['AAPL', 'DIS', 'XOM', 'INTC', 'MSFT', 'AMZN', 'NVDA', 'CRM', 'GOOGL', 'TSLA']

        df = pd.read_csv(csv_path)
        df.set_index('datadate', inplace=True)
        df = df[tickers].dropna()
        data = df.to_numpy(dtype='float32')
        data_tensor = torch.FloatTensor(data)

        rolled = rolling_window(data_tensor, window_size)
        self.samples = pct_change(rolled)  # shape: (N, window_size-1, D)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx]

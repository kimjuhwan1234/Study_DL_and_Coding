import os

import pandas as pd
import yfinance as yf
from fbm import MBM
import tqdm
import torch
import numpy as np

from .utils import sample_indices, load_obj, save_obj


def train_test_split(
        x: torch.Tensor,
        train_test_ratio: float,
        device: str
):
    size = x.shape[0]
    train_set_size = int(size * train_test_ratio)

    indices_train = sample_indices(size, train_set_size, device)
    indices_test = torch.LongTensor([i for i in range(size) if i not in indices_train])

    x_train = x[indices_train]
    x_test = x[indices_test]
    return x_train, x_test


def download_stock_price(
        ticker: str,
        start: str = '2012-01-01',
        end: str = '2025-01-01',
        interval: str = '1d',
):
    '''
    If you want to download other stock data, please do it before execute your code.
    '''
    dataframe = yf.download(ticker, start=start, end=end, interval=interval)
    file_name = ticker + "_" + interval + ".csv"
    csv_file_file = os.path.join("datasets", "stock", file_name)
    if not os.path.exists(csv_file_file):
        dataframe.to_csv(file_name)
    return dataframe


def rolling_window(x: torch.Tensor, window_size: int):
    '''
    See https://se.mathworks.com/help/econ/rolling-window-estimation-of-state-space-models.html
    '''
    print("Tensor shape before rolling:", x.shape)
    windowed_data = []
    for t in range(x.shape[0] - window_size + 1):
        window = x[t:t + window_size, :]
        windowed_data.append(window)
    print("Tensor shape after rolling:", torch.stack(windowed_data, dim=0).shape)
    return torch.stack(windowed_data, dim=0)


def transfer_percentage(x):
    '''
    Calculate the percentage change of each element in the sequence relative to its starting value, 
    ignoring sequences that start with a zero value.
    '''
    start = x[:, 0:1, :]

    # remove zero start
    idx_ = torch.nonzero(start == 0, as_tuple=False).tolist()
    if idx_:
        idx_ = idx_[0]
    idx_ = list(set(list(range(x.shape[0]))) - set(idx_))

    new_x = x[idx_, ...]
    new_start = start[idx_, ...]

    new_x = (new_x - new_start) / new_start
    return new_x


def pct_change(x: torch.Tensor):
    '''
    Calculate percent change along time dimension (dim=1) just like pandas pct_change.
    Returns a tensor of shape (N, T-1, D).
    '''
    # x: shape (N, T, D)
    prev = x[:, :-1, :]
    curr = x[:, 1:, :]

    # Avoid division by zero
    zero_mask = prev == 0
    pct = (curr - prev) / prev
    pct[zero_mask] = 0  # or torch.nan, depending on your needs

    return pct


def get_stock_price(data_config):
    """
    Get stock price
    Returns
    -------
    dataset: torch.Tensor
        torch.tensor of shape (#data, window_size, 1)
    """
    csv_file_name = data_config['ticker'] + "_" + data_config['interval'] + ".csv"
    pt_file_name = data_config['ticker'] + "_" + data_config['interval'] + "_rolled.pt"
    csv_file_path = os.path.join(data_config['dir'], data_config['subdir'], csv_file_name)
    pt_file_path = os.path.join(data_config['dir'], data_config['subdir'], pt_file_name)

    if not os.path.exists(csv_file_name):
        _ = download_stock_price(ticker=data_config['ticker'], interval=data_config['interval'])

    if os.path.exists(pt_file_path):
        dataset = load_obj(pt_file_path)
        print(f'Rolled data for training, shape {dataset.shape}')

    else:
        df = pd.read_csv(csv_file_path)
        print(f'Original data: {os.path.basename(csv_file_name)}, shape {df.shape}')
        dataset = df[df.columns[data_config['column']]].to_numpy(dtype='float')
        dataset = torch.FloatTensor(dataset).unsqueeze(dim=1)
        # print(dataset[:5])
        dataset = rolling_window(dataset, data_config['window_size'])
        dataset = transfer_percentage(dataset)

        print(f'Rolled data for training, shape {dataset.shape}')
        save_obj(dataset, pt_file_path)
    return dataset


def get_rBergomi_paths(hurst=0.25, size=2200, n_lags=100, maturity=1, xi=0.5, eta=0.5):
    r"""
    Paths of Rough stochastic volatility model for an asset price process S_t of the form

    dS_t = \sqrt(V_t) S_t dZ_t
    V_t := \xi * exp(\eta * W_t^H - 0.5*\eta^2*t^{2H})

    where W_t^H denotes the Riemann-Liouville fBM given by

    W_t^H := \int_0^t K(t-s) dW_t,  K(r) := \sqrt{2H} r^{H-1/2}

    with W_t,Z_t correlated brownian motions (I'm actually considering \rho=0)

    Parameters
    ----------
    hurst: float,
    size: int
        size of the dataset
    n_lags: int
        Number of timesteps in the path
    maturity: float
        Final time. Should be a value in [0,1]
    xi: float
    eta: float

    Returns
    -------
    dataset: np.array
        array of shape (size, n_lags, 2)

    """
    assert hurst < 0.5, "hurst parameter should be < 0.5"

    dataset = np.zeros((size, n_lags, 2))

    for j in tqdm(range(size), total=size):
        m = MBM(n=n_lags - 1, hurst=lambda t: hurst, length=maturity, method='riemannliouville')
        fbm = m.mbm()  # fractional Brownian motion
        times = m.times()
        V = xi * np.exp(eta * fbm - 0.5 * eta ** 2 * times ** (2 * hurst))

        h = times[1:] - times[:-1]  # time increments
        brownian_increments = np.random.randn(h.shape[0]) * np.sqrt(h)

        log_S = np.zeros_like(V)
        log_S[1:] = (-0.5 * V[:-1] * h + np.sqrt(
            V[:-1]) * brownian_increments).cumsum()  # Ito formula to get SDE for  d log(S_t). We assume S_0 = 1
        S = np.exp(log_S)
        dataset[j] = np.stack([S, V], 1)
    return dataset


def get_gbm(size, n_lags, d=1, drift=0., scale=0.1, h=1):
    '''
    See Wiki: https://en.wikipedia.org/wiki/Geometric_Brownian_motion#Simulating_sample_paths
    '''
    S_t = torch.ones(size, n_lags, d)
    S_t[:, 1:, :] = (drift - scale ** 2 / 2) * h + torch.normal(mean=0, std=np.sqrt(h), size=(size, n_lags - 1, d))
    S_t = torch.exp(S_t)
    S_t = S_t.cumprod(1)
    return S_t

"""
Estimate the thresholds for producing trading signals based on training data
"""

import numpy as np
import pandas as pd

import os
from os.path import *

import torch
from torch import nn

from .Transform import Tensor, Inc2Price, movingaverage

your_path = 'data/'
parent_data_path = your_path

def windowize(data: torch.Tensor, window: int = 60):
    T_total, N = data.shape
    B = T_total - window + 1
    windows = torch.stack([data[i:i+window] for i in range(B)])  # (B, T, N)
    return windows


def gen_thresholds(data_name, tickers, strategy, percentile_l, WH):
    thresholds_data_folder = join(your_path, 'Thresholds', data_name)
    os.makedirs(thresholds_data_folder, exist_ok=True)

    if 'MR' in strategy and 'Port' not in strategy:
        thresholds_path = join(thresholds_data_folder, '_'.join(tickers) + '_MR_%s.npy' % '_'.join(map(str, percentile_l)))
    elif 'MR' in strategy and 'Port' in strategy:
        thresholds_path = join(thresholds_data_folder, '_'.join(tickers) + '_Port*MR_%s.npy' % '_'.join(map(str, percentile_l)))
    elif 'TF' in strategy and 'Port' not in strategy:
        thresholds_path = join(thresholds_data_folder, '_'.join(tickers) + '_TF_%s.npy' % '_'.join(map(str, percentile_l)))
    elif 'TF' in strategy and 'Port' in strategy:
        thresholds_path = join(thresholds_data_folder, '_'.join(tickers) + '__Port*TF_%s.npy' % '_'.join(map(str, percentile_l)))
    else:
        pass

    if isfile(thresholds_path):
        thresholds_array_stocks = np.load(thresholds_path)
    else:
        csv_path = join(parent_data_path, 'sp500.csv')
        df = pd.read_csv(csv_path)
        df.set_index('datadate', inplace=True)
        df = df[tickers].dropna()
        data_l = df.to_numpy(dtype='float32')
        data = Tensor(data_l)

        prices_l = windowize(data)


        # 1. B, T, N -> (B*T, N)
        prices_l_flat = prices_l.reshape(-1, prices_l.shape[2])  # (B*T, N)

        thresholds_array_list = []

        if 'MR' in strategy:
            # 2. 평균: 최근 WH시점에 대해 시간축(dim=1) 기준 이동 평균
            prices_l_ma = torch.mean(prices_l[:, -WH:, :], dim=1, keepdim=True)  # (B, 1, N)
            prices_l_ma = prices_l_ma.expand(-1, prices_l.shape[1], -1)  # (B, T, N)
            prices_l_ma_flat = prices_l_ma.reshape(-1, prices_l.shape[2])  # (B*T, N)

            # 3. Z-score 계산
            zscores = (prices_l_flat - prices_l_ma_flat) / 0.01  # (B*T, N)
            zscores_np = zscores.cpu().detach().numpy()

            # 4. 각 종목별 threshold 계산
            for i in range(zscores_np.shape[1]):  # for each stock
                thresholds = np.percentile(zscores_np[:, i], percentile_l)
                thresholds_array_list.append(thresholds)

        elif 'TF' in strategy:
            # 2. 이동 평균 계산 (dim=1 기준 시간축)
            prices_l = prices_l.transpose(2, 1)
            prices_l_ma = movingaverage(prices_l, WH)  # (B, T, N)
            prices_l_ma2 = movingaverage(prices_l, WH*2)  # (B, T, N)

            prices_l_ma_flat = prices_l_ma.reshape(-1, prices_l.shape[2])  # (B*T, N)
            prices_l_ma2_flat = prices_l_ma2.reshape(-1, prices_l.shape[2])  # (B*T, N)

            # 3. Z-score 계산
            zscores = (prices_l_ma_flat - prices_l_ma2_flat) / 0.01
            zscores_np = zscores.cpu().detach().numpy()

            # 4. 각 종목별 threshold 계산
            for i in range(zscores_np.shape[1]):
                thresholds = np.percentile(zscores_np[:, i], percentile_l)
                thresholds_array_list.append(thresholds)

        # 5. (N, len(percentile_l)) 형태
        thresholds_array_stocks = np.stack(thresholds_array_list)

        np.save(thresholds_path, thresholds_array_stocks)
    return thresholds_array_stocks
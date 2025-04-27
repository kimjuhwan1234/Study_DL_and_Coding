import os
import numpy as np
import pandas as pd
import torch
from .datasets import rolling_window

class Normalization():
    """ Standard scales a given (indexed) input vector along the specified axis. """

    def __init__(self, axis=(1)):
        self.mean = None
        self.std = None
        self.axis = axis

    def normalize(self, x):
        if self.mean is None:
            self.mean = torch.mean(x, dim=self.axis)
            self.std = torch.std(x, dim=self.axis)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def rescale(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)
    
 
def get_var_dataset(window_size, batch_size=5000, dim=3, phi=0.8, sigma=0.8):
    def VAR(window_size, dim=3, phi=0.8, sigma=0.5, burn_in=200):
        window_size = window_size + burn_in
        Xt = np.zeros((window_size, dim))
        one = np.ones(dim)
        identity = np.identity(dim)
        mu = np.zeros(dim)
        cov = sigma * one + (1 - sigma) * identity

        # The epsilon term in (17)
        E = np.random.multivariate_normal(mu, cov, window_size)
        for i in range(dim):
            Xt[0, i] = 0
        for t in range(window_size - 1):
            Xt[t + 1] = phi * Xt[t] + E[t]
        return Xt[burn_in:]

    var_samples = []
    for i in range(batch_size):
        sample = VAR(window_size, dim, phi=phi, sigma=sigma)
        var_samples.append(sample)
    raw_data = torch.from_numpy(np.array(var_samples)).float()

    transform = Normalization()
    normalized_data = transform.normalize(raw_data)

    return raw_data, normalized_data

def get_data(data_type, p, q, **data_params):
    if data_type == 'VAR':
         _, normalized_data = get_var_dataset(
            40000, batch_size=1, **data_params
        )
    assert normalized_data.shape[0] == 1
    normalized_data = rolling_window(normalized_data[0], p + q)
    return normalized_data
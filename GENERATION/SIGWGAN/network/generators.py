import torch
import signatory
import torch.nn as nn
from .arfnn import ResidualNN, FeedForwardNN
from ..utils.augmentations import apply_augmentations, get_number_of_channels_after_augmentations


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')


def compute_multilevel_logsignature(brownian_path, time_brownian, time_u, time_t, depth):
    """Compute multi-level log-signature features for given time intervals."""
    u_indices = torch.searchsorted(time_brownian, time_u, right=True) - 1
    t_indices = torch.searchsorted(time_brownian, time_t[1:], right=True) - 1
    u_idx_for_t = torch.searchsorted(time_u, time_t[1:], right=True) - 1

    u_logsigrnn, multi_level_log_sig, last_u_idx = [], [], -1

    for idx_t, idx_u in zip(t_indices, u_idx_for_t):
        idx_u = max(idx_u, 0)
        if idx_u != last_u_idx:
            u_logsigrnn.append(time_u[idx_u])
            last_u_idx = idx_u
        interval = brownian_path[:, u_indices[idx_u]:idx_t + 1]
        multi_level_log_sig.append(signatory.logsignature(interval, depth=depth, basepoint=True))

    return [torch.zeros_like(multi_level_log_sig[0])] + multi_level_log_sig, u_logsigrnn


class LSTMGenerator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, n_layers: int, init_fixed: bool = True):
        super().__init__()
        self.input_dim, self.output_dim, self.init_fixed = input_dim, output_dim, init_fixed
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)
        self.linear.apply(init_weights)

    def forward(self, batch_size: int, window_size: int, device: str) -> torch.Tensor:
        z = 0.1 * torch.randn(batch_size, window_size, self.input_dim, device=device)
        z[:, 0] = 0
        z = z.cumsum(1)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size,
                         device=device) if self.init_fixed else None
        c0 = torch.zeros_like(h0)
        h1, _ = self.lstm(z, (h0, c0))
        x = self.linear(h1)
        assert x.shape[1] == window_size
        return x


class LogSigRNNGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, augmentations, depth, hidden_dim, len_noise=1000, len_interval_u=50,
                 init_fixed=True):
        super().__init__()
        self.depth, self.augmentations, self.input_dim, self.output_dim, self.hidden_dim, self.init_fixed = depth, augmentations, input_dim, output_dim, hidden_dim, init_fixed

        input_dim_rnn = get_number_of_channels_after_augmentations(input_dim, augmentations)
        logsig_channels = signatory.logsignature_channels(input_dim_rnn, depth)

        self.len_noise = len_noise
        self.time_brownian = torch.linspace(0, 1, len_noise)
        self.time_u = self.time_brownian[::len_interval_u]

        self.rnn = nn.Sequential(
            FeedForwardNN(hidden_dim + logsig_channels, hidden_dim, [hidden_dim, hidden_dim]),
            nn.Tanh()
        )
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)
        self.rnn.apply(init_weights)
        self.linear.apply(init_weights)

        self.initial_nn = nn.Sequential(
            ResidualNN(input_dim, hidden_dim, [hidden_dim, hidden_dim]),
            nn.Tanh()
        )
        self.initial_nn.apply(init_weights)

    def forward(self, batch_size: int, window_size: int, device: str):
        time_t = torch.linspace(0, 1, window_size, device=device)
        z = torch.randn(batch_size, self.len_noise, self.input_dim, device=device)
        h = (self.time_brownian[1:] - self.time_brownian[:-1]).reshape(1, -1, 1).repeat(batch_size, 1,
                                                                                        self.input_dim).to(device)
        z[:, 1:] *= torch.sqrt(h)
        z[:, 0] = 0
        brownian_path = z.cumsum(1)

        y = apply_augmentations(brownian_path, self.augmentations) if self.augmentations else brownian_path
        y_logsig, u_logsigrnn = compute_multilevel_logsignature(y, self.time_brownian.to(device),
                                                                self.time_u.to(device), time_t, self.depth)
        u_logsigrnn.append(time_t[-1])

        h0 = torch.zeros(batch_size, self.hidden_dim, device=device) if self.init_fixed else self.initial_nn(
            torch.randn(batch_size, self.input_dim, device=device))
        last_h, x = h0, torch.zeros(batch_size, window_size, self.output_dim, device=device)

        for idx, (t, y_logsig_) in enumerate(zip(time_t, y_logsig)):
            h = self.rnn(torch.cat([last_h, y_logsig_], dim=-1))
            if t >= u_logsigrnn[0]:
                del u_logsigrnn[0]
                last_h = h
            x[:, idx] = self.linear(h)

        assert x.shape[1] == window_size
        return x

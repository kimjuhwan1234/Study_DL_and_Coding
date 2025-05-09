import torch
import signatory
import torch.nn as nn

from ..network.arfnn import ResidualNN, FeedForwardNN
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


class LogSigRNNGenerator(nn.Module):
    def __init__(self, input_dim, augmentations, depth, window_size, hidden_dim, device,
                 len_noise=1000, len_interval_u=50,
                 init_fixed=True):
        super().__init__()
        self.depth, self.augmentations, self.input_dim, self.hidden_dim, self.init_fixed, self.device, self.window_size = depth, augmentations, input_dim, hidden_dim, init_fixed, device, window_size

        input_dim_rnn = get_number_of_channels_after_augmentations(input_dim, augmentations)
        logsig_channels = signatory.logsignature_channels(input_dim_rnn, depth)

        self.len_noise = len_noise
        self.time_brownian = torch.linspace(0, 1, len_noise)
        self.time_u = self.time_brownian[::len_interval_u]

        self.rnn = nn.Sequential(
            FeedForwardNN(hidden_dim + logsig_channels, hidden_dim, [hidden_dim, hidden_dim]),
            nn.Tanh()
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rnn.apply(init_weights)
        self.linear.apply(init_weights)

        self.initial_nn = nn.Sequential(
            ResidualNN(input_dim, hidden_dim, [hidden_dim, hidden_dim]),
            nn.Tanh()
        )
        self.initial_nn.apply(init_weights)

    def forward(self, batch_size):
        time_t = torch.linspace(0, 1, self.window_size, device=self.device)
        z = torch.randn(batch_size, self.len_noise, self.input_dim, device=self.device)
        h = (self.time_brownian[1:] - self.time_brownian[:-1]).reshape(1, -1, 1).repeat(batch_size, 1,
                                                                                        self.input_dim).to(self.device)
        z[:, 1:] *= torch.sqrt(h)
        z[:, 0] = 0
        brownian_path = z.cumsum(1)

        y = apply_augmentations(brownian_path, self.augmentations) if self.augmentations else brownian_path
        y_logsig, u_logsigrnn = compute_multilevel_logsignature(y, self.time_brownian.to(self.device),
                                                                self.time_u.to(self.device), time_t, self.depth)
        u_logsigrnn.append(time_t[-1])

        h0 = torch.zeros(batch_size, self.hidden_dim, device=self.device) if self.init_fixed else self.initial_nn(
            torch.randn(batch_size, self.input_dim, device=self.device))
        last_h, x = h0, torch.zeros(batch_size, self.window_size, self.hidden_dim, device=self.device)

        for idx, (t, y_logsig_) in enumerate(zip(time_t, y_logsig)):
            h = self.rnn(torch.cat([last_h, y_logsig_], dim=-1))
            if t >= u_logsigrnn[0]:
                del u_logsigrnn[0]
                last_h = h
            x[:, idx] = self.linear(h)

        assert x.shape[1] == self.window_size
        return x


class LogSigRNNEncoder(nn.Module):
    def __init__(self, input_dim, augmentations, depth, hidden_dim, batch_size, window_size, device,
                 len_interval_u=50,
                 init_fixed=True):
        super().__init__()
        self.depth, self.augmentations, self.input_dim, self.hidden_dim, self.init_fixed, self.batch_size, self.window_size, self.device = depth, augmentations, input_dim, hidden_dim, init_fixed, batch_size, window_size, device
        input_dim_rnn = get_number_of_channels_after_augmentations(input_dim, augmentations)
        logsig_channels = signatory.logsignature_channels(input_dim_rnn, depth)

        self.time_brownian = torch.linspace(0, 1, window_size)
        self.time_u = self.time_brownian[::len_interval_u]

        self.rnn = nn.Sequential(
            FeedForwardNN(hidden_dim + logsig_channels, hidden_dim, [hidden_dim, hidden_dim]),
            nn.Tanh()
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rnn.apply(init_weights)
        self.linear.apply(init_weights)

        self.initial_nn = nn.Sequential(
            ResidualNN(input_dim, hidden_dim, [hidden_dim, hidden_dim]),
            nn.Tanh()
        )
        self.initial_nn.apply(init_weights)

    def forward(self, real_data):
        time_t = torch.linspace(0, 1, self.window_size, device=self.device)

        # Assume real_data is already on the correct device and shaped properly
        y = apply_augmentations(real_data, self.augmentations) if self.augmentations else real_data
        y_logsig, u_logsigrnn = compute_multilevel_logsignature(
            y, self.time_brownian.to(self.device), self.time_u.to(self.device), time_t, self.depth
        )
        u_logsigrnn.append(time_t[-1])

        h0 = torch.zeros(self.batch_size, self.hidden_dim, device=self.device) if self.init_fixed else self.initial_nn(
            real_data[:, 0]  # Assuming first time step feature used for init
        )
        last_h = h0
        x = torch.zeros(self.batch_size, self.window_size, self.hidden_dim, device=self.device)

        for idx, (t, y_logsig_) in enumerate(zip(time_t, y_logsig)):
            h = self.rnn(torch.cat([last_h, y_logsig_], dim=-1))
            if t >= u_logsigrnn[0]:
                del u_logsigrnn[0]
                last_h = h
            x[:, idx] = self.linear(h)

        assert x.shape[1] == self.window_size
        return x

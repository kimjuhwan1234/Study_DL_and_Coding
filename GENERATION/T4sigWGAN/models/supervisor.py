import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def make_rnn_cell(rnn_type, input_dim, hidden_dim):
    if rnn_type == 'GRU':
        return nn.GRU(input_dim, hidden_dim, batch_first=True)
    elif rnn_type == 'LSTM':
        return nn.LSTM(input_dim, hidden_dim, batch_first=True)
    else:
        raise ValueError("Unsupported RNN type")


class Supervisor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.rnn = make_rnn_cell('GRU', hidden_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, h, lengths):
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        return self.fc(out)


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation,
                     groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv',
                      get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
    result.add_module('bn', get_bn(out_channels))
    return result


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1:, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight[None, None, :] + self.affine_bias[None, None, :]
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias[None, None, :]) / self.affine_weight[None, None, :]
        x = x * self.stdev + self.mean
        return x


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel, small_kernel_merged=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups,
                                         bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups)
            if small_kernel is not None:
                self.small_conv = conv_bn(in_channels, out_channels, small_kernel, stride, small_kernel // 2, groups)

    def forward(self, x):
        if hasattr(self, 'lkb_reparam'):
            return self.lkb_reparam(x)
        out = self.lkb_origin(x)
        if hasattr(self, 'small_conv'):
            out += self.small_conv(x)
        return out


# class Flatten_Head(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.linear = nn.Linear(d_model, d_model)
#
#     def forward(self, x):         # x: [B, C, T]
#         x = x.permute(0, 2, 1)    # → [B, T, C]
#         x = self.linear(x)        # → [B, T, 1]
#         return x


class ModernTCN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.revin = RevIN(configs.enc_in, affine=configs.affine) if configs.revin else None
        self.decomp = series_decomp(configs.kernel_size) if configs.decomposition else None

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        c_in = configs.enc_in
        for i in range(len(configs.dims)):
            conv = ReparamLargeKernelConv(c_in, configs.dims[i],
                                          kernel_size=configs.large_size[i],
                                          stride=1,
                                          groups=1,
                                          small_kernel=configs.small_size[i],
                                          small_kernel_merged=configs.small_kernel_merged)
            self.conv_layers.append(conv)
            self.norm_layers.append(nn.BatchNorm1d(configs.dims[i]))
            c_in = configs.dims[i]

        # self.head = Flatten_Head(configs.dims[-1])

    def forward(self, x):  # x: [B, T, C]
        if self.revin:
            x = self.revin(x, 'norm')
        if self.decomp:
            x, _ = self.decomp(x)
        x = x.permute(0, 2, 1)  # [B, C, T]
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x = conv(x)
            x = norm(x)
            x = F.relu(x)
        out = x.transpose(1, 2)
        return out

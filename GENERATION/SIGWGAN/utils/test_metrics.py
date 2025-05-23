from functools import partial
from typing import Tuple, Optional

from .utils import to_numpy
from ..sigw1metric import SigW1Metric

import torch
from torch import nn
import numpy as np


def cov_torch(x, rowvar=False, bias=True, ddof=None, aweights=None):
    # """Estimates covariance matrix like numpy.cov"""
    ## reshape x
    # _, T, C = x.shape
    # x = x.reshape(-1, T * C)
    ## ensure at least 2D
    # if x.dim() == 1:
    #    x = x.view(-1, 1)

    ## treat each column as a data point, each row as a variable
    # if rowvar and x.shape[0] != 1:
    #    x = x.t()

    # if ddof is None:
    #    if bias == 0:
    #        ddof = 1
    #    else:
    #        ddof = 0

    # w = aweights
    # if w is not None:
    #    if not torch.is_tensor(w):
    #        w = torch.tensor(w, dtype=torch.float)
    #    w_sum = torch.sum(w)
    #    avg = torch.sum(x * (w / w_sum)[:, None], 0)
    # else:
    #    avg = torch.mean(x, 0)

    ## Determine the normalization
    # if w is None:
    #    fact = x.shape[0] - ddof
    # elif ddof == 0:
    #    fact = w_sum
    # elif aweights is None:
    #    fact = w_sum - ddof
    # else:
    #    fact = w_sum - ddof * torch.sum(w * w) / w_sum

    # xm = x.sub(avg.expand_as(x))

    # if w is None:
    #    X_T = xm.t()
    # else:
    #    X_T = torch.mm(torch.diag(w), xm).t()

    # c = torch.mm(X_T, xm)
    # c = c / fact

    # return c.squeeze()
    device = x.device
    x = to_numpy(x)
    _, L, C = x.shape
    x = x.reshape(-1, L * C)
    return torch.from_numpy(np.cov(x, rowvar=False)).to(device).float()


def acf_torch(x: torch.Tensor, max_lag: int, dim: Tuple[int] = (0, 1)) -> torch.Tensor:
    """
    :param x: torch.Tensor [B, S, D]
    :param max_lag: int. specifies number of lags to compute the acf for
    :return: acf of x. [max_lag, D]
    """
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def cacf_torch(x, max_lag, dim=(0, 1)):
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    ind = get_lower_triangular_indices(x.shape[2])
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    cacf_list = list()
    for i in range(max_lag):
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
        cacf_i = torch.mean(y, (1))
        cacf_list.append(cacf_i)
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def skew_torch(x, dim=(0, 1), dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_3 = torch.pow(x, 3).mean(dim, keepdims=True)
    x_std_3 = torch.pow(x.std(dim, unbiased=True, keepdims=True), 3)
    skew = x_3 / x_std_3
    if dropdims:
        skew = skew[0, 0]
    return skew


def kurtosis_torch(x, dim=(0, 1), excess=True, dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    kurtosis = x_4 / x_var2
    if excess:
        kurtosis = kurtosis - 3
    if dropdims:
        kurtosis = kurtosis[0, 0]
    return kurtosis


class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


acf_diff = lambda x: torch.sqrt(torch.pow(x, 2).sum(0))
cc_diff = lambda x: torch.abs(x).sum(0)


def cov_diff(x): return torch.abs(x).mean()


class ACFLoss(Loss):
    def __init__(self, x_real, max_lag=64, **kwargs):
        super(ACFLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.acf_real = acf_torch(self.transform(x_real), max_lag, dim=(0, 1))
        self.max_lag = max_lag

    def compute(self, x_fake):
        acf_fake = acf_torch(self.transform(x_fake), self.max_lag)
        return self.norm_foo(acf_fake - self.acf_real.to(x_fake.device))


class MeanLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(MeanLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.mean = x_real.mean((0, 1))

    def compute(self, x_fake, **kwargs):
        return self.norm_foo(x_fake.mean((0, 1)) - self.mean)


class StdLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(StdLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.std_real = x_real.std((0, 1))

    def compute(self, x_fake, **kwargs):
        return self.norm_foo(x_fake.std((0, 1)) - self.std_real)


class SkewnessLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(SkewnessLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.skew_real = skew_torch(self.transform(x_real))

    def compute(self, x_fake, **kwargs):
        skew_fake = skew_torch(self.transform(x_fake))
        return self.norm_foo(skew_fake - self.skew_real)


class KurtosisLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(KurtosisLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.kurtosis_real = kurtosis_torch(self.transform(x_real))

    def compute(self, x_fake):
        kurtosis_fake = kurtosis_torch(self.transform(x_fake))
        return self.norm_foo(kurtosis_fake - self.kurtosis_real)


class CrossCorrelLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(CrossCorrelLoss, self).__init__(norm_foo=cc_diff, **kwargs)
        self.cross_correl_real = cacf_torch(self.transform(x_real), 1).mean(0)[0]

    def compute(self, x_fake):
        cross_correl_fake = cacf_torch(self.transform(x_fake), 1).mean(0)[0]
        loss = self.norm_foo(cross_correl_fake - self.cross_correl_real.to(x_fake.device)).unsqueeze(0)
        return loss


import torch
from torch import nn


def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    b = b + 1e-5 if b == a else b
    # delta = (b - a) / n_bins
    bins = torch.linspace(a, b, n_bins + 1)
    delta = bins[1] - bins[0]
    # bins = torch.arange(a, b + 1.5e-5, step=delta)
    count = torch.histc(x, bins=n_bins, min=a, max=b).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


class HistoLoss(Loss):

    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            tmp_densities = list()
            tmp_locs = list()
            tmp_deltas = list()
            for t in range(x_real.shape[1]):
                x_ti = x_real[:, t, i].reshape(-1, 1)
                d, b = histogram_torch(x_ti, n_bins, density=True)
                tmp_densities.append(nn.Parameter(d).to(x_real.device))
                delta = b[1:2] - b[:1]
                loc = 0.5 * (b[1:] + b[:-1])
                tmp_locs.append(loc)
                tmp_deltas.append(delta)
            self.densities.append(tmp_densities)
            self.locs.append(tmp_locs)
            self.deltas.append(tmp_deltas)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            tmp_loss = list()
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1).to(x_fake.device)
                x_ti = x_fake[:, t, i].contiguous(
                ).view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                counter = (relu(self.deltas[i][t].to(
                    x_fake.device) / 2. - dist) > 0.).float()
                density = counter.mean(0) / self.deltas[i][t].to(x_fake.device)
                abs_metric = torch.abs(
                    density - self.densities[i][t].to(x_fake.device))
                loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise


class CovLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(CovLoss, self).__init__(norm_foo=cov_diff, **kwargs)
        self.covariance_real = cov_torch(
            self.transform(x_real))

    def compute(self, x_fake):
        covariance_fake = cov_torch(self.transform(x_fake))
        loss = self.norm_foo(covariance_fake -
                             self.covariance_real.to(x_fake.device))
        return loss


class SigW1Loss(Loss):
    def __init__(self, x_real, **kwargs):
        name = kwargs.pop('name')
        super(SigW1Loss, self).__init__(name=name)
        self.sig_w1_metric = SigW1Metric(x_real=x_real, **kwargs)

    def compute(self, x_fake):
        loss = self.sig_w1_metric(x_fake)
        return loss


diff = lambda x: x[:, 1:] - x[:, :-1]
test_metrics = {
    'acf_abs': partial(ACFLoss, name='acf_abs', transform=torch.abs),
    'acf_id': partial(ACFLoss, name='acf_id'),
    'acf_id_rtn': partial(ACFLoss, name='acf_id_rtn', transform=diff),
    'abs_metric': partial(HistoLoss, n_bins=50, name='abs_metric'),
    'kurtosis': partial(KurtosisLoss, name='kurtosis'),
    'kurtosis_rtn': partial(KurtosisLoss, name='kurtosis_rtn', transform=diff),
    'skew': partial(SkewnessLoss, name='skew'),
    'skew_rtn': partial(SkewnessLoss, name='skew_rtn', transform=diff),
    'mean': partial(MeanLoss, name='mean'),
    'std': partial(StdLoss, name='std'),
    'cross_correl': partial(CrossCorrelLoss, name='cross_correl'),
    'cross_correl_rtn': partial(CrossCorrelLoss, name='cross_correl_rtn', transform=diff),
    'covariance': partial(CovLoss, name='covariance', transform=torch.exp),
    'covariance_rtn': partial(CovLoss, name='covariance_rtn', transform=diff),
    'sig_w1': partial(SigW1Loss, name='sig_w1', augmentations=[], normalise=False, mask_rate=0.01, depth=4)
}


def is_multivariate(x: torch.Tensor):
    """ Check if the path / tensor is multivariate. """
    return True if x.shape[-1] > 1 else False


def get_standard_test_metrics(x: torch.Tensor, augmentations: Tuple = ()):
    """ Initialise list of standard test metrics for evaluating the goodness of the generator. """
    test_metrics_list = [
        # test_metrics['abs_metric'](x),
        # test_metrics['acf_id'](x, max_lag=2),
        # test_metrics['acf_id_rtn'](x, max_lag=2),
        # test_metrics['skew'](x),
        # test_metrics['kurtosis'](x),
        # test_metrics['skew_rtn'](x),
        # test_metrics['kurtosis_rtn'](x),
        # test_metrics['covariance'](x, reg=1),
        # test_metrics['covariance_rtn'](x, reg=1),

        # Original: test_metrics['sig_w1'](x)
        # Omit augmentations
        partial(SigW1Loss, name='sig_w1', augmentations=augmentations, normalise=False, mask_rate=0.01, depth=4)(x)
    ]
    # if is_multivariate(x):
    #     test_metrics_list.append(test_metrics['cross_correl'](x))
    #     test_metrics_list.append(test_metrics['cross_correl_rtn'](x))
    return test_metrics_list

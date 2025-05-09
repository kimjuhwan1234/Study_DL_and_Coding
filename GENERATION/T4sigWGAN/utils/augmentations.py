from typing import List, Tuple
from dataclasses import dataclass

import torch
import signatory


def get_time_vector(batch_size: int, length: int) -> torch.Tensor:
    return torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(batch_size, 1, 1)


def lead_lag_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation. 
    See "A Primer on the Signature Method in Machine Learning"
    """
    # 0-th dim is batch
    repeat = torch.repeat_interleave(x, repeats=2, dim=1)
    lead_lag = torch.cat([repeat[:, :-1], repeat[:, 1:]], dim=2)
    return lead_lag


def lead_lag_transform_with_time(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation for a multivariate paths.
    """
    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    t_repeat = torch.repeat_interleave(t, repeats=3, dim=1)
    x_repeat = torch.repeat_interleave(x, repeats=3, dim=1)
    time_lead_lag = torch.cat([
        t_repeat[:, 0:-2],
        x_repeat[:, 1:-1],
        x_repeat[:, 2:],
    ], dim=2)
    return time_lead_lag


def sig_normal(sig: torch.Tensor, normalize=False):
    if normalize == False:
        return sig.mean(0)
    elif normalize == True:
        mu = sig.mean(0)
        sigma = sig.std(0)
        sig = (sig - mu) / sigma
        return sig


def I_visibility_transform(path: torch.Tensor) -> torch.Tensor:
    '''
    Implement using definition from 'Signature features with the visibility transformation'.
    Return 
        torch.Tensor of shape (N,L+2,d+1)

    x1_I: zero tensor of shape (N,1,d+1)
    x2_I: (x1,0) of shape (N,1,d+1)
    xk_I: (xk,1) of shape (N,1,d+1)
    
    Note that for stock csv data, the x0 data is stored at the first element, not the last one.
    '''
    x1 = torch.zeros_like(path[:, :1, :])
    x2 = path[:, :1, :]
    first_two_rows = torch.cat([x1, x2], dim=1)

    path_add_rows = torch.cat([first_two_rows, path], dim=1)

    appended_zeros = torch.zeros_like(path[:, :2, :1])
    appended_ones = torch.ones_like(path[:, :, :1])
    appended = torch.cat([appended_zeros, appended_ones], dim=1)

    output = torch.cat([path_add_rows, appended], dim=-1)
    return output


def T_visibility_transform(path: torch.Tensor) -> torch.Tensor:
    '''
    Implement using definition from 'Signature features with the visibility transformation'.
    Return 
        torch.Tensor of shape (N,L+2,d+1)

    xn+2_I: zero tensor of shape (N,1,d+1)
    xn+1_I: (xn,0) of shape (N,1,d+1)
    xk_I: (xk,1) of shape (N,1,d+1)
    
    Note that for stock csv data, the xn data is stored at the last element, not the first one.
    '''
    xlast = torch.zeros_like(path[:, -1:, :])
    xlast_ = path[:, -1:, :]
    last_two_rows = torch.cat([xlast_, xlast], dim=1)

    path_add_rows = torch.cat([path, last_two_rows], dim=1)

    appended_zeros = torch.zeros_like(path[:, -2:, :1])
    appended_ones = torch.ones_like(path[:, :, :1])
    appended = torch.cat([appended_ones, appended_zeros], dim=1)

    output = torch.cat([path_add_rows, appended], dim=-1)
    return output


def get_number_of_channels_after_augmentations(input_dim, augmentations):
    x = torch.zeros(1, 10, input_dim)
    y = apply_augmentations(x, augmentations)
    return y.shape[-1]


@dataclass
class BaseAugmentation:
    pass

    def apply(self, *args: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError('Needs to be implemented by child.')


@dataclass
class AddTime(BaseAugmentation):

    def apply(self, x: torch.Tensor):
        t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
        return torch.cat([t, x], dim=-1)


@dataclass
class LeadLag(BaseAugmentation):
    with_time: bool = False

    def apply(self, x: torch.Tensor):
        if self.with_time:
            return lead_lag_transform_with_time(x)
        else:
            return lead_lag_transform(x)


@dataclass
class VisiTrans(BaseAugmentation):
    type: str = "I"

    def apply(self, x: torch.Tensor):
        if self.type == "I":
            return I_visibility_transform(x)
        elif self.type == "T":
            return T_visibility_transform(x)


@dataclass
class Cumsum(BaseAugmentation):
    '''
    See 'A Primer on the Signature Method in Machine Learning' 2.1.1
    '''
    dim: int = 1

    def apply(self, x: torch.Tensor):
        return x.cumsum(dim=self.dim)


def apply_augmentations(x: torch.Tensor, augmentations: Tuple) -> torch.Tensor:
    y = x.clone()
    for augmentation in augmentations:
        y = augmentation.apply(y)
    return y


def augment_path_and_compute_signatures(x: torch.Tensor, config: dict) -> torch.Tensor:
    y = apply_augmentations(x, config["augmentations"])
    return signatory.signature(y, config["depth"], basepoint=False)


AUGMENTATIONS = {'AddTime': AddTime, 'LeadLag': LeadLag, 'VisiTrans': VisiTrans, 'CumSum': Cumsum}


def parse_augmentations(list_of_dicts):
    augmentations = list()
    for kwargs in list_of_dicts:
        name = kwargs.pop('name')
        augmentations.append(
            AUGMENTATIONS[name](**kwargs)
        )
    return augmentations

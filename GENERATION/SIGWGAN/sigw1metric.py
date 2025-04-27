import math
import torch
import signatory
from typing import Tuple, Optional

from .utils.augmentations import apply_augmentations


def rmse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).pow(2).sum().sqrt()


def masked_rmse(x: torch.Tensor, y: torch.Tensor, mask_rate: float, device: str) -> torch.Tensor:
    mask = (torch.rand(x.size(0), device=device) > mask_rate).float()
    return ((x - y).pow(2) * mask.unsqueeze(1)).mean().sqrt()


def compute_expected_signature(
        x_path: torch.Tensor, depth: int, augmentations: Tuple = (), normalise: bool = False
) -> torch.Tensor:
    x_aug = apply_augmentations(x_path, augmentations)
    expected_signature = signatory.signature(x_aug, depth=depth).mean(0)

    if normalise:
        dim, count = x_aug.size(2), 0
        for i in range(depth):
            degree = dim ** (i + 1)
            expected_signature[count:count + degree] *= math.factorial(i + 1)
            count += degree

    return expected_signature


class SigW1Metric:
    def __init__(
            self, depth: int, x_real: torch.Tensor, mask_rate: float,
            augmentations: Optional[Tuple] = (), normalise: bool = False
    ):
        assert x_real.ndim == 3, f'Path must be 3D. Got {x_real.ndim}D.'

        self.depth = depth
        self.window_size = x_real.size(1)
        self.mask_rate = mask_rate
        self.augmentations = augmentations
        self.normalise = normalise

        self.expected_signature_mu = compute_expected_signature(
            x_real, depth, augmentations, normalise
        )

    def __call__(self, x_path_nu: torch.Tensor) -> torch.Tensor:
        expected_signature_nu = compute_expected_signature(
            x_path_nu, self.depth, self.augmentations, self.normalise
        )
        return rmse(self.expected_signature_mu.to(x_path_nu.device), expected_signature_nu)

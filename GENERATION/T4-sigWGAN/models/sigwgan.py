import torch
from torch import nn
from tqdm import tqdm
from ..utils.utils import to_numpy
from collections import defaultdict
from ..sigw1metric import SigW1Metric
from ..utils.augmentations import augment_path_and_compute_signatures
from ..utils.augmentations import SignatureConfig
def sigcwgan_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    return torch.norm(sig_pred - sig_fake_conditional_expectation, p=2, dim=1).mean()


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



@dataclass
class SigCWGANConfig:
    mc_size: int
    sig_config_future: SignatureConfig
    sig_config_past: SignatureConfig

    def compute_sig_past(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_past)

    def compute_sig_future(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_future)


def calibrate_sigw1_metric(config, x_future, x_past):
    sigs_past = config.compute_sig_past(x_past)
    sigs_future = config.compute_sig_future(x_future)
    assert sigs_past.size(0) == sigs_future.size(0)
    X, Y = to_numpy(sigs_past), to_numpy(sigs_future)
    lm = LinearRegression()
    lm.fit(X, Y)
    sigs_pred = torch.from_numpy(lm.predict(X)).float().to(x_future.device)
    return sigs_pred


def sample_sig_fake(G, q, sig_config, x_past):
    x_past_mc = x_past.repeat(sig_config.mc_size, 1, 1).requires_grad_()
    x_fake = G.sample(q, x_past_mc)
    sigs_fake_future = sig_config.compute_sig_future(x_fake)
    sigs_fake_ce = sigs_fake_future.reshape(sig_config.mc_size, x_past.size(0), -1).mean(0)
    return sigs_fake_ce, x_fake




class SigWGAN(nn.Module):
    def __init__(self, G, lr_generator, epoch, batch_size, x_real_rolled,
                 test_metrics_test, **kwargs):
        super().__init__()
        self.G = G
        self.epoch = epoch
        self.batch_size = batch_size
        self.test_metrics_test = test_metrics_test

        self.x_real_rolled = x_real_rolled

        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr_generator)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.G_optimizer, gamma=0.95, step_size=128)
        self.losses_history = defaultdict(list)

    def fit(self, device):
        self.G.to(device)
        best_loss = float('inf')
        pbar = tqdm(range(self.epoch), desc='Training')

        for epoch_idx in pbar:
            self.G_optimizer.zero_grad()

            x_fake = self.G(
                batch_size=self.batch_size,
                window_size=self.sig_w1_metric.window_size,
                device=device
            )
            sigs_fake_future = sig_config.compute_sig_past(self.x_real_rolled)
            sigs_fake_future = sig_config.compute_sig_future(x_fake)
            sigs_fake_ce = sigs_fake_future.reshape(sig_config.mc_size, x_past.size(0), -1).mean(0)
            loss = sigcwgan_loss(sigs_fake_ce)
            loss.backward()
            best_loss = loss.item() if epoch_idx == 0 else best_loss
            self.G_optimizer.step()
            self.scheduler.step()

            self.losses_history['sig_w1_loss'].append(loss.item())
            pbar.set_description(f"sig-w1 loss: {loss.item():.4f}")

            self.evaluate(x_fake)

    def evaluate(self, x_fake):
        with torch.no_grad():
            for metric in self.test_metrics_test:
                metric(x_fake)
                loss = to_numpy(metric.loss_componentwise)
                if loss.ndim == 1:
                    loss = loss[..., None]
                self.losses_history[f'{metric.name}_test'].append(loss)
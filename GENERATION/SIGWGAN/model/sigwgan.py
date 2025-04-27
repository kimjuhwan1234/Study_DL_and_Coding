import torch
from torch import nn
from tqdm import tqdm
from ..utils.utils import to_numpy
from collections import defaultdict
from ..sigw1metric import SigW1Metric


class SigWGAN(nn.Module):
    def __init__(self, G, lr_generator, epoch, batch_size, depth, x_real_rolled, augmentations,
                 test_metrics_test, normalise_sig: bool = True, mask_rate=0.01, **kwargs):
        super().__init__()
        self.G = G
        self.epoch = epoch
        self.batch_size = batch_size
        self.test_metrics_test = test_metrics_test

        self.sig_w1_metric = SigW1Metric(
            depth=depth, x_real=x_real_rolled, augmentations=augmentations,
            mask_rate=mask_rate, normalise=normalise_sig
        )

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

            loss = self.sig_w1_metric(x_fake)
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
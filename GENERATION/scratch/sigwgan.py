# from dataclasses import dataclass
# from collections import defaultdict
#
# import torch
# from torch import nn
# from tqdm import tqdm
#
# from ..utils.utils import to_numpy
# from ..utils.augmentations import SignatureConfig
# from ..utils.augmentations import augment_path_and_compute_signatures
#
#
# def sigcwgan_loss(sig_real: torch.Tensor, sig_fake: torch.Tensor):
#     return torch.norm(sig_real - sig_fake, p=2, dim=1).mean()
#
#
# @dataclass
# class SigCWGANConfig:
#     mc_size: int
#     sig_config_future: SignatureConfig
#     sig_config_past: SignatureConfig
#
#     def compute_sig_past(self, x):
#         return augment_path_and_compute_signatures(x, self.sig_config_past)
#
#     def compute_sig_future(self, x):
#         return augment_path_and_compute_signatures(x, self.sig_config_future)
#
#
# class SigWGAN(nn.Module):
#     def __init__(self, G, lr_generator, epoch, batch_size, x_real_rolled,
#                  test_metrics_test, sig_config, **kwargs):
#         super().__init__()
#         self.G = G
#         self.epoch = epoch
#         self.batch_size = batch_size
#         self.sig_config = sig_config
#         self.test_metrics_test = test_metrics_test
#
#         self.x_real_rolled = x_real_rolled
#
#         self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr_generator)
#         self.scheduler = torch.optim.lr_scheduler.StepLR(self.G_optimizer, gamma=0.95, step_size=128)
#         self.losses_history = defaultdict(list)
#
#     def fit(self, device):
#         self.G.to(device)
#         best_loss = float('inf')
#         pbar = tqdm(range(self.epoch), desc='Training')
#
#         for epoch_idx in pbar:
#             self.G_optimizer.zero_grad()
#
#             x_fake = self.G(
#                 batch_size=self.batch_size,
#                 window_size=self.sig_w1_metric.window_size,
#                 device=device
#             )
#             sigs_real = self.sig_config.compute_sig_past(self.x_real_rolled)
#             embdding = nn.Linear(sigs_real.shape[1], x_fake.shape[1])
#             x_real = embdding(sigs_real)
#             loss = sigcwgan_loss(x_real, x_fake)
#
#             loss.backward()
#             best_loss = loss.item() if epoch_idx == 0 else best_loss
#             self.G_optimizer.step()
#             self.scheduler.step()
#
#             self.losses_history['sig_w1_loss'].append(loss.item())
#             pbar.set_description(f"sig-w1 loss: {loss.item():.4f}")
#
#             self.evaluate(x_fake)
#
#     def evaluate(self, x_fake):
#         with torch.no_grad():
#             for metric in self.test_metrics_test:
#                 metric(x_fake)
#                 loss = to_numpy(metric.loss_componentwise)
#                 if loss.ndim == 1:
#                     loss = loss[..., None]
#                 self.losses_history[f'{metric.name}_test'].append(loss)

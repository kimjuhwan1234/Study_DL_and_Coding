import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TimeGAN(nn.Module):
    def __init__(self, Encoder, Decoder, Generator, Supervisor, Discriminator):
        super().__init__()
        self.embedder = Encoder
        self.recovery = Decoder
        self.generator = Generator
        self.supervisor = Supervisor
        self.discriminator = Discriminator

    def forward(self, x):
        h = self.embedder(x)
        x_tilde = self.recovery(h)

        e_hat = self.generator()
        h_hat = self.supervisor(e_hat)
        h_hat_supervise = self.supervisor(h)

        x_hat = self.recovery(h_hat)

        return {
            'x_tilde': x_tilde, 'x_hat': x_hat,
            'h': h, 'h_hat': h_hat, 'h_hat_supervise': h_hat_supervise,
            'e_hat': e_hat,
        }


def mse(x, y):
    return torch.mean((x - y) ** 2)


def train_T4sigWGAN(model, ori_data, ori_lengths, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Extract params
    batch_size = params['batch_size']
    iterations = params['iterations']
    gamma = params.get('gamma', 1.0)

    optimizer_e = optim.Adam(list(model.embedder.parameters()) + list(model.recovery.parameters()))
    optimizer_g = optim.Adam(list(model.generator.parameters()) + list(model.supervisor.parameters()))

    ori_data = torch.tensor(ori_data, dtype=torch.float32).to(device)
    ori_lengths = torch.tensor(ori_lengths, dtype=torch.int64).to(device)
    dataset = TensorDataset(ori_data, ori_lengths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for step in range(iterations):
        for x_mb, len_mb in loader:
            model.train()
            # ===== Embedding network =====
            h = model.embedder(x_mb, len_mb)
            x_tilde = model.recovery(h, len_mb)
            e_loss = mse(x_mb, x_tilde)
            optimizer_e.zero_grad()
            e_loss.backward()
            optimizer_e.step()

        if step % 500 == 0:
            print(f"[Embedder] Step {step}, Loss: {e_loss.item():.4f}")

    for step in range(iterations):
        for x_mb, len_mb in loader:
            # ===== Supervised loss only (G + S) =====
            h = model.embedder(x_mb, len_mb)
            h_hat_supervise = model.supervisor(h, len_mb)
            s_loss = mse(h[:, 1:, :], h_hat_supervise[:, :-1, :])
            optimizer_g.zero_grad()
            s_loss.backward()
            optimizer_g.step()

        if step % 500 == 0:
            print(f"[Supervisor] Step {step}, Loss: {s_loss.item():.4f}")

    for step in range(iterations):
        for x_mb, len_mb in loader:
            z_mb = torch.randn_like(x_mb).to(device)
            model.train()

            out = model(x_mb, z_mb, len_mb)

            # Generator losses
            g_loss_u = nn.BCEWithLogitsLoss()(out['y_fake'], torch.ones_like(out['y_fake']))
            g_loss_ue = nn.BCEWithLogitsLoss()(out['y_fake_e'], torch.ones_like(out['y_fake_e']))
            g_loss_s = mse(out['h'][:, 1:, :], out['h_hat_supervise'][:, :-1, :])

            # Moment matching
            g_loss_v1 = torch.mean(torch.abs(torch.std(out['x_hat'], dim=0) - torch.std(x_mb, dim=0)))
            g_loss_v2 = torch.mean(torch.abs(torch.mean(out['x_hat'], dim=0) - torch.mean(x_mb, dim=0)))
            g_loss_v = g_loss_v1 + g_loss_v2

            g_loss = g_loss_u + gamma * g_loss_ue + 100 * torch.sqrt(g_loss_s) + 100 * g_loss_v

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

        if step % 500 == 0:
            print(
                f"[Joint] Step {step}, G_adv: {g_loss_u.item():.4f}, G_sup: {g_loss_s.item():.4f}, G_mom: {g_loss_v.item():.4f}")

    print("[Training Finished]")

    return model

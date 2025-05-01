import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Helper: RNN wrapper
def make_rnn_cell(rnn_type, input_dim, hidden_dim):
    if rnn_type == 'GRU':
        return nn.GRU(input_dim, hidden_dim, batch_first=True)
    elif rnn_type == 'LSTM':
        return nn.LSTM(input_dim, hidden_dim, batch_first=True)
    else:
        raise ValueError("Unsupported RNN type")

class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = make_rnn_cell('GRU', input_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h_packed, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h_packed, batch_first=True)
        return self.fc(h)

class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.rnn = make_rnn_cell('GRU', hidden_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, h, lengths):
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        return self.fc(out)

class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.rnn = make_rnn_cell('GRU', z_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, z, lengths):
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        return self.fc(out)

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

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.rnn = make_rnn_cell('GRU', hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h, lengths):
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        return self.fc(out)


class TimeGAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.embedder = Embedder(input_dim, hidden_dim, 1)
        self.recovery = Recovery(hidden_dim, input_dim)
        self.generator = Generator(z_dim, hidden_dim)
        self.supervisor = Supervisor(hidden_dim)
        self.discriminator = Discriminator(hidden_dim)

    def forward(self, x, z, lengths):
        h = self.embedder(x, lengths)
        x_tilde = self.recovery(h, lengths)

        e_hat = self.generator(z, lengths)
        h_hat = self.supervisor(e_hat, lengths)
        h_hat_supervise = self.supervisor(h, lengths)

        x_hat = self.recovery(h_hat, lengths)

        y_real = self.discriminator(h, lengths)
        y_fake = self.discriminator(h_hat.detach(), lengths)
        y_fake_e = self.discriminator(e_hat.detach(), lengths)

        return {
            'x_tilde': x_tilde, 'x_hat': x_hat,
            'h': h, 'h_hat': h_hat, 'h_hat_supervise': h_hat_supervise,
            'e_hat': e_hat,
            'y_real': y_real, 'y_fake': y_fake, 'y_fake_e': y_fake_e
        }

def mse(x, y):
    return torch.mean((x - y) ** 2)

def train_timegan(model, ori_data, ori_lengths, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Extract params
    batch_size = params['batch_size']
    iterations = params['iterations']
    gamma = params.get('gamma', 1.0)

    optimizer_e = optim.Adam(list(model.embedder.parameters()) + list(model.recovery.parameters()))
    optimizer_g = optim.Adam(list(model.generator.parameters()) + list(model.supervisor.parameters()))
    optimizer_d = optim.Adam(model.discriminator.parameters())

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
            s_loss = mse(h[:,1:,:], h_hat_supervise[:,:-1,:])
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
            g_loss_s = mse(out['h'][:,1:,:], out['h_hat_supervise'][:,:-1,:])

            # Moment matching
            g_loss_v1 = torch.mean(torch.abs(torch.std(out['x_hat'], dim=0) - torch.std(x_mb, dim=0)))
            g_loss_v2 = torch.mean(torch.abs(torch.mean(out['x_hat'], dim=0) - torch.mean(x_mb, dim=0)))
            g_loss_v = g_loss_v1 + g_loss_v2

            g_loss = g_loss_u + gamma * g_loss_ue + 100 * torch.sqrt(g_loss_s) + 100 * g_loss_v

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            # Discriminator
            out = model(x_mb, z_mb, len_mb)
            d_loss_real = nn.BCEWithLogitsLoss()(out['y_real'], torch.ones_like(out['y_real']))
            d_loss_fake = nn.BCEWithLogitsLoss()(out['y_fake'].detach(), torch.zeros_like(out['y_fake']))
            d_loss_fake_e = nn.BCEWithLogitsLoss()(out['y_fake_e'].detach(), torch.zeros_like(out['y_fake_e']))
            d_loss = d_loss_real + d_loss_fake + gamma * d_loss_fake_e

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

        if step % 500 == 0:
            print(f"[Joint] Step {step}, D: {d_loss.item():.4f}, G_adv: {g_loss_u.item():.4f}, G_sup: {g_loss_s.item():.4f}, G_mom: {g_loss_v.item():.4f}")

    print("[Training Finished]")

    return model
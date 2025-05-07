from ..utils.loss import *


class T4sigWGAN(nn.Module):
    def __init__(self, Encoder, Decoder, Generator, Supervisor, Discriminator):
        super().__init__()
        self.embedder = Encoder
        self.recovery = Decoder
        self.generator = Generator
        self.supervisor = Supervisor
        self.discriminator = Discriminator

    def forward(self, x, stage):

        if stage == 'Pretrain_1':
            h = self.embedder(x)
            x_tilde = self.recovery(h)
            loss = mse(x, x_tilde)
            return x_tilde, loss

        elif stage == 'Pretrain_2':
            h = self.embedder(x)
            h_hat_supervise = self.supervisor(h)
            loss = sigcwgan_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])
            x_tilde = self.recovery(h)
            return x_tilde, loss

        else:
            e_hat = self.generator()
            h_hat = self.supervisor(e_hat)
            h = self.embedder(x)
            loss = sigcwgan_loss(h, h_hat)
            x_hat = self.recovery(h_hat)
            return x_hat, loss

# g_loss_u = nn.BCEWithLogitsLoss()(out['y_fake'], torch.ones_like(out['y_fake']))
# g_loss_ue = nn.BCEWithLogitsLoss()(out['y_fake_e'], torch.ones_like(out['y_fake_e']))
# g_loss_s = mse(out['h'][:, 1:, :], out['h_hat_supervise'][:, :-1, :])
# g_loss = g_loss_u + gamma * g_loss_ue + 100 * torch.sqrt(g_loss_s)

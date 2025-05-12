from ..utils.loss import *


class T4sigWGAN(nn.Module):
    def __init__(self, Encoder, Decoder, Generator, Supervisor, Discriminator, batch_size):
        super().__init__()
        self.embedder = Encoder
        self.recovery = Decoder
        self.generator = Generator
        self.supervisor = Supervisor
        self.discriminator = Discriminator
        self.batch_size = batch_size
        self.criterion = Score()

    def forward(self, stage, x=None):

        if x is not None:
            if stage == 'Pretrain_1':
                h = self.embedder(x)
                x_tilde = self.recovery(h)
                loss = rmse(x, x_tilde)
                return x_tilde, loss

            elif stage == 'Pretrain_2':
                h = self.embedder(x)
                h_hat_supervise = self.supervisor(h)
                loss = sigcwgan_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])
                x_tilde = self.recovery(h)
                return x_tilde, loss

            else:
                e_hat = self.generator(self.batch_size)
                h_hat = self.supervisor(e_hat)
                h = self.embedder(x)
                loss = sigcwgan_loss(h, h_hat)
                x_hat = self.recovery(h_hat)

                PNL, PNL_validity = self.discriminator(x)
                gen_PNL, gen_PNL_validity = self.discriminator(x_hat)

                fake_score = self.criterion(gen_PNL_validity, PNL)
                real_score = self.criterion(PNL_validity, PNL)
                loss_G = fake_score
                loss_D = real_score - fake_score


                return x_hat, loss, loss_G, loss_D

        else:
            e_hat = self.generator(stage)
            h_hat = self.supervisor(e_hat)
            x_hat = self.recovery(h_hat)
            return x_hat

# g_loss_u = nn.BCEWithLogitsLoss()(out['y_fake'], torch.ones_like(out['y_fake']))
# g_loss_ue = nn.BCEWithLogitsLoss()(out['y_fake_e'], torch.ones_like(out['y_fake_e']))
# g_loss_s = mse(out['h'][:, 1:, :], out['h_hat_supervise'][:, :-1, :])
# g_loss = g_loss_u + gamma * g_loss_ue + 100 * torch.sqrt(g_loss_s)

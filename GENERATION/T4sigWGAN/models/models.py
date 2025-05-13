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
                x_tilde = self.recovery(h)
                h_hat_supervise = self.supervisor(h)
                loss = sigcwgan_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])
                return x_tilde, loss

            else:
                e_hat = self.generator(self.batch_size)
                h = self.embedder(x)

                h_hat = self.supervisor(e_hat)
                h_hat_supervise = self.supervisor(h)

                x_hat = self.recovery(h_hat)
                x_tilde = self.recovery(h)

                PNL, PNL_validity = self.discriminator(x)
                gen_PNL, gen_PNL_validity = self.discriminator(x_hat)

                loss_r = rmse(x, x_tilde)
                loss_s = sigcwgan_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])
                loss_u = sigcwgan_loss(h, h_hat)

                fake_score = self.criterion(gen_PNL_validity, PNL)
                real_score = self.criterion(PNL_validity, PNL)

                loss_G = fake_score / 100000 + loss_u
                loss_D = (real_score - fake_score) / 100000
                loss_SR = loss_s + loss_r

                return x_hat, loss_G, loss_D, loss_SR

        else:
            e_hat = self.generator(stage)
            h_hat = self.supervisor(e_hat)
            x_hat = self.recovery(h_hat)
            return x_hat

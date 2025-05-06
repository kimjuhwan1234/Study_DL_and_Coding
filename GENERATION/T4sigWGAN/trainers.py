import tqdm
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from itertools import chain
from .utils.utils import plot_hist
from .utils.loss import *


class Trainer:
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
    ):
        self.args = args
        self.device = args.device

        self.train_hist = pd.DataFrame(columns=['epoch', 'loss', 'acc'])
        self.val_hist = pd.DataFrame(columns=['epoch', 'loss', 'acc'])

        self.model = model
        self.model.to(self.device)
        self.criterion = Score()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.submission_dataloader = submission_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.E0optim = Adam(
            chain(self.model.embedder.parameters(), self.model.recovery.parameters()),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.Goptim = Adam(
            chain(self.model.generator.parameters(), self.model.supervisor.parameters(),),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.Eoptim = Adam(
            self.model.recovery.final.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.Soptim = Adam(
            self.model.supervisor.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.Doptim = Adam(
            self.model.discriminator.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.lr_schedulerE0 = ReduceLROnPlateau(self.E0optim, mode='min', factor=0.2, patience=10)
        self.lr_schedulerG = ReduceLROnPlateau(self.Goptim, mode='min', factor=0.2, patience=10)
        self.lr_schedulerE = ReduceLROnPlateau(self.Eoptim, mode='min', factor=0.2, patience=10)
        self.lr_schedulerS = ReduceLROnPlateau(self.Soptim, mode='min', factor=0.2, patience=10)
        self.lr_schedulerD = ReduceLROnPlateau(self.Doptim, mode='min', factor=0.2, patience=10)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, mode="valid")

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, mode="test")

    def submission(self, epoch):
        return self.iteration(epoch, self.submission_dataloader, mode="submission")

    def iteration(self, epoch, dataloader, mode="train"):
        raise NotImplementedError

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def get_lr(self, opt):
        for param_group in opt.param_groups:
            return param_group['lr']

    def metric(self, output, gt):
        """R-squared (R²) 계산"""
        _, predicted = torch.max(output, 1)
        correct = (predicted == gt).sum().item()
        return correct

    def evaluate(self):
        self.train_hist = self.train_hist.map(
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)
        self.val_hist = self.val_hist.map(
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)

        loss_df = pd.concat([self.train_hist['epoch'], self.train_hist['loss'], self.val_hist['loss']], axis=1)
        loss_df.set_index(['epoch'], inplace=True)
        acc_df = pd.concat([self.train_hist['epoch'], self.train_hist['acc'], self.val_hist['acc']], axis=1)
        acc_df.set_index(['epoch'], inplace=True)
        plot_hist(loss_df.index, loss_df, 'Loss')
        plot_hist(acc_df.index, acc_df, 'R2')


class ERTrainer(Trainer):
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
    ):
        super(ERTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def iteration(self, epoch, dataloader, mode="train"):

        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(
            enumerate(dataloader),
            desc=f"{self.args.model_name} EP_{mode}:{epoch}",
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        if mode == "train":
            self.model.train()
            total_loss = 0.0
            cur_loss = 0.0
            total_acc = 0.0
            total = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                input = batch.to(self.device)

                # Binary cross_entropy
                h = self.model.embedder(input)
                x_tilde = self.model.recovery(h)
                loss = mse(input, x_tilde)
                acc = mae(input, x_tilde)
                self.E0optim.zero_grad()
                loss.backward()
                self.E0optim.step()

                total_loss += loss
                cur_loss = loss
                total_acc += acc
                total += batch.size(0)

            avg_loss = total_loss / len(rec_data_iter)
            avg_acc = 100 * total_acc / total

            new_data = pd.DataFrame([[epoch, avg_loss, avg_acc]], columns=["epoch", "loss", "acc"])
            self.train_hist = pd.concat([self.train_hist.astype("float32"), new_data.astype("float32")],
                                        axis=0).reset_index(drop=True)

            post_fix = {
                "epoch": epoch,
                "avg_loss": "{:.4f}".format(total_loss / len(rec_data_iter)),
                "cur_loss": "{:.4f}".format(cur_loss),
                "current lr": "{:.4f}".format(self.get_lr(self.E0optim)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

        else:
            self.model.eval()
            total_loss = 0.0
            total_acc = 0.0
            total = 0.0

            batch_results = []
            with torch.no_grad():
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or CPU)
                    input = batch.to(self.device)

                    # Binary cross_entropy
                    h = self.model.embedder(input)
                    x_tilde = self.model.recovery(h)
                    loss = mse(input, x_tilde)
                    acc = mae(input, x_tilde)
                    self.E0optim.zero_grad()
                    loss.backward()
                    self.E0optim.step()

                    total_loss += loss
                    total_acc += acc
                    total += batch.size(0)

            avg_loss = total_loss / len(rec_data_iter)
            avg_acc = 100 * total_acc / total

            new_data = pd.DataFrame([[epoch, avg_loss, avg_acc]], columns=["epoch", "loss", "acc"])
            self.val_hist = pd.concat([self.val_hist.astype("float32"), new_data.astype("float32")],
                                      axis=0).reset_index(drop=True)

            self.lr_schedulerE0.step(avg_loss)
            return avg_loss.cpu().detach().numpy()


class GSTrainer(Trainer):
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
    ):
        super(GSTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def iteration(self, epoch, dataloader, mode="train"):

        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(
            enumerate(dataloader),
            desc=f"{self.args.model_name} EP_{mode}:{epoch}",
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        if mode == "train":
            self.model.train()
            total_loss = 0.0
            cur_loss = 0.0
            total_acc = 0.0
            total = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                input = batch.to(self.device)

                # Binary cross_entropy
                h = self.model.embedder(input)
                h_hat_supervise = self.model.supervisor(h)
                loss = sigcwgan_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])
                acc = mse(h[:, 1:, :], h_hat_supervise[:, :-1, :])
                self.Soptim.zero_grad()
                loss.backward()
                self.Soptim.step()

                total_loss += loss
                cur_loss = loss
                total_acc += acc
                total += batch.size(0)

            avg_loss = total_loss / len(rec_data_iter)
            avg_acc = 100 * total_acc / total

            new_data = pd.DataFrame([[epoch, avg_loss, avg_acc]], columns=["epoch", "loss", "acc"])
            self.train_hist = pd.concat([self.train_hist.astype("float32"), new_data.astype("float32")],
                                        axis=0).reset_index(drop=True)

            post_fix = {
                "epoch": epoch,
                "avg_loss": "{:.4f}".format(total_loss / len(rec_data_iter)),
                "cur_loss": "{:.4f}".format(cur_loss),
                "current lr": "{:.4f}".format(self.get_lr(self.Soptim)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

        else:
            self.model.eval()
            total_loss = 0.0
            total_acc = 0.0
            total = 0.0

            batch_results = []
            with torch.no_grad():
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or CPU)
                    input = batch.to(self.device)

                    # Binary cross_entropy
                    h = self.model.embedder(input)
                    h_hat_supervise = self.model.supervisor(h)
                    loss = sigcwgan_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])
                    acc = mse(h[:, 1:, :], h_hat_supervise[:, :-1, :])
                    self.Soptim.zero_grad()
                    loss.backward()
                    self.Soptim.step()

                    total_loss += loss
                    total_acc += acc
                    total += batch.size(0)

            avg_loss = total_loss / len(rec_data_iter)
            avg_acc = 100 * total_acc / total

            new_data = pd.DataFrame([[epoch, avg_loss, avg_acc]], columns=["epoch", "loss", "acc"])
            self.val_hist = pd.concat([self.val_hist.astype("float32"), new_data.astype("float32")],
                                      axis=0).reset_index(drop=True)

            self.lr_schedulerS.step(avg_loss)
            return avg_loss.cpu().detach().numpy()


class FinetuneTrainer(Trainer):
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
    ):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def iteration(self, epoch, dataloader, mode="train"):

        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(
            enumerate(dataloader),
            desc=f"{self.args.model_name} EP_{mode}:{epoch}",
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        if mode == "train":
            self.model.train()
            total_loss = 0.0
            cur_loss = 0.0
            total_acc = 0.0
            total = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                input = batch.to(self.device)

                # Binary cross_entropy
                h_hat = self.model.generator()
                h = self.model.embedder(input)
                x_fake = self.model.recovery(h_hat)
                loss = sigcwgan_loss(h, h_hat)

                acc = mae(h, h_hat)
                self.Goptim.zero_grad()
                loss.backward()
                self.Goptim.step()

                # Adversarial loss
                PNL, PNL_validity = self.model.discriminator(input)
                gen_PNL, gen_PNL_validity = self.model.discriminator(x_fake)
                real_score = self.criterion(PNL_validity, PNL)
                fake_score = self.criterion(gen_PNL_validity, PNL)
                loss_D = real_score - fake_score
                loss_G = fake_score

                self.Doptim.zero_grad()
                loss_D.backward()
                self.Doptim.step()

                self.Eoptim.zero_grad()
                loss_G.backward()
                self.Eoptim.step()

                total_loss += loss
                cur_loss = loss
                total_acc += acc
                total += batch.size(0)

            avg_loss = total_loss / len(rec_data_iter)
            avg_acc = 100 * total_acc / total

            new_data = pd.DataFrame([[epoch, avg_loss, avg_acc]], columns=["epoch", "loss", "acc"])
            self.train_hist = pd.concat([self.train_hist.astype("float32"), new_data.astype("float32")],
                                        axis=0).reset_index(drop=True)

            post_fix = {
                "epoch": epoch,
                "avg_loss": "{:.4f}".format(total_loss / len(rec_data_iter)),
                "cur_loss": "{:.4f}".format(cur_loss),
                "current lr": "{:.4f}".format(self.get_lr(self.optim)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

        else:
            self.model.eval()
            total_loss = 0.0
            total_acc = 0.0
            total = 0.0

            batch_results = []
            with torch.no_grad():
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or CPU)
                    input = batch.to(self.device)

                    # Binary cross_entropy
                    h_hat = self.model.generator()
                    h= self.model.embedder(input)
                    loss = sigcwgan_loss(h, h_hat)

                    acc = mae(h, h_hat)
                    self.Goptim.zero_grad()
                    loss.backward()
                    self.Goptim.step()

                    total_loss += loss
                    total_acc += acc
                    total += batch.size(0)

            avg_loss = total_loss / len(rec_data_iter)
            avg_acc = 100 * total_acc / total

            new_data = pd.DataFrame([[epoch, avg_loss, avg_acc]], columns=["epoch", "loss", "acc"])
            self.val_hist = pd.concat([self.val_hist.astype("float32"), new_data.astype("float32")],
                                      axis=0).reset_index(drop=True)

            self.lr_schedulerG.step(avg_loss)
            return avg_loss.cpu().detach().numpy()
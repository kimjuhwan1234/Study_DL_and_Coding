import tqdm
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils.loss import *
from .utils.utils import plot_hist


class Trainer:
    def __init__(self, args, model, train_dataloader, eval_dataloader):
        self.args = args
        self.model = model
        self.device = args.device
        self.model.to(self.device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.file_name = args.ROOT_DIR + args.data_name + args.model_name + ".pt"



        self.train_hist = pd.DataFrame(columns=['epoch', 'loss', 'acc'])
        self.val_hist = pd.DataFrame(columns=['epoch', 'loss', 'acc'])

        self.lr_dict = {}
        self.optim_dict = {}
        betas = (self.args.adam_beta1, self.args.adam_beta2)

        self.optim_dict['E'] = Adam(
            list(self.model.embedder.parameters()) + list(self.model.recovery.parameters()),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.optim_dict['G'] = Adam(
            list(self.model.generator.parameters()) + list(self.model.supervisor.parameters()),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.optim_dict['R'] = Adam(
            self.model.recovery.final.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.optim_dict['S'] = Adam(
            list(self.model.generator.parameters()) + list(self.model.supervisor.parameters()),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.optim_dict['D'] = Adam(
            self.model.discriminator.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.lr_dict['E'] = ReduceLROnPlateau(self.optim_dict['E'], mode='min', factor=0.2, patience=10)
        self.lr_dict['G'] = ReduceLROnPlateau(self.optim_dict['G'], mode='min', factor=0.2, patience=10)
        self.lr_dict['R'] = ReduceLROnPlateau(self.optim_dict['R'], mode='min', factor=0.2, patience=10)
        self.lr_dict['S'] = ReduceLROnPlateau(self.optim_dict['S'], mode='min', factor=0.2, patience=10)
        self.lr_dict['D'] = ReduceLROnPlateau(self.optim_dict['D'], mode='min', factor=0.2, patience=10)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch, stage):
        self.iteration(epoch, stage, self.train_dataloader)

    def valid(self, epoch, stage):
        return self.iteration(epoch, stage, self.eval_dataloader, mode="valid")

    def iteration(self, epoch, stage, dataloader, mode="train"):
        raise NotImplementedError

    def save(self):
        torch.save(self.model.cpu().state_dict(), self.file_name)

    def load(self):
        self.model.load_state_dict(torch.load(self.file_name))
        self.model.to(self.device)

    def metric(self, output, gt):
        """R-squared (R²) 계산"""
        _, predicted = torch.max(output, 1)
        correct = (predicted == gt).sum().item()
        return correct

    def evaluate(self, title):
        self.train_hist = self.train_hist.applymap(
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)
        self.val_hist = self.val_hist.applymap(
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)

        loss_df = pd.concat([self.train_hist['epoch'], self.train_hist['loss'], self.val_hist['loss']], axis=1)
        loss_df.set_index(['epoch'], inplace=True)
        acc_df = pd.concat([self.train_hist['epoch'], self.train_hist['acc'], self.val_hist['acc']], axis=1)
        acc_df.set_index(['epoch'], inplace=True)
        plot_hist(loss_df.index, loss_df, title)
        # plot_hist(acc_df.index, acc_df, 'R2')

        self.train_hist = pd.DataFrame(columns=['epoch', 'loss', 'acc'])
        # self.val_hist = pd.DataFrame(columns=['epoch', 'loss', 'acc'])


class FinetuneTrainer(Trainer):
    def __init__(self, args, model, train_dataloader, eval_dataloader):
        super(FinetuneTrainer, self).__init__(args, model, train_dataloader, eval_dataloader)

    def iteration(self, epoch, stage, dataloader, mode="train"):

        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(
            enumerate(dataloader),
            desc=f"{self.args.model_name} EP_{mode}:{epoch}",
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        if mode == "train":
            self.model.train()
            cur_loss = 0.0
            total_loss = 0.0
            total_acc = 0.0
            total = 0.0

            for i, batch in rec_data_iter:
                input = batch.to(self.device)
                if stage == "Pretrain_1":
                    x_tilde, loss = self.model(stage, input)
                    self.optim_dict['E'].zero_grad()
                    loss.backward()
                    self.optim_dict['E'].step()

                elif stage == "Pretrain_2":
                    x_tilde, loss = self.model(stage, input)
                    self.optim_dict['S'].zero_grad()
                    loss.backward()
                    self.optim_dict['S'].step()

                else:
                    x_hat, loss, loss_G, loss_D = self.model(stage, input)
                    # PNL, PNL_validity = self.model.discriminator(input)
                    # gen_PNL, gen_PNL_validity = self.model.discriminator(x_hat)
                    #
                    # fake_score = self.criterion(gen_PNL_validity, PNL)
                    # real_score = self.criterion(PNL_validity, PNL)
                    # loss_G = fake_score
                    # loss_D = real_score - fake_score

                    self.optim_dict['R'].zero_grad()
                    loss_G.backward(retain_graph=True)
                    self.optim_dict['R'].step()

                    self.optim_dict['D'].zero_grad()
                    loss_D.backward(retain_graph=True)
                    self.optim_dict['D'].step()

                    self.optim_dict['G'].zero_grad()
                    loss.backward()
                    self.optim_dict['G'].step()



                cur_loss = loss
                total_loss += loss
                total_acc += 0
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
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

        else:
            self.model.eval()
            total_loss = 0.0
            total_acc = 0.0
            total = 0.0

            with torch.no_grad():
                for i, batch in rec_data_iter:
                    input = batch.to(self.device)
                    if stage == "Pretrain_1":
                        x_tilde, loss = self.model(stage, input)

                    elif stage == "Pretrain_2":
                        x_tilde, loss = self.model(stage, input)

                    else:
                        x_hat, loss, loss_G, loss_D = self.model(stage, input)
                        # PNL, PNL_validity = self.model.discriminator(input)
                        # gen_PNL, gen_PNL_validity = self.model.discriminator(x_hat)
                        # real_score = self.criterion(PNL_validity, PNL)
                        # fake_score = self.criterion(gen_PNL_validity, PNL)
                        # loss_D = real_score - fake_score
                        # loss_G = fake_score

                    total_loss += loss
                    total_acc += 0
                    total += batch.size(0)

            avg_loss = total_loss / len(rec_data_iter)
            avg_acc = 100 * total_acc / total

            new_data = pd.DataFrame([[epoch, avg_loss, avg_acc]], columns=["epoch", "loss", "acc"])
            self.val_hist = pd.concat([self.val_hist.astype("float32"), new_data.astype("float32")],
                                      axis=0).reset_index(drop=True)

            if stage == "Pretrain_1":
                self.lr_dict['E'].step(avg_loss)


            elif stage == "Pretrain_2":
                self.lr_dict['S'].step(avg_loss)

            else:
                self.lr_dict['R'].step(avg_loss)
                self.lr_dict['G'].step(avg_loss)
                self.lr_dict['D'].step(avg_loss)


            return avg_loss.cpu().detach().numpy()

from tqdm import tqdm
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils.loss import *
from .utils.utils import plot_hist, plot_tail


class Trainer:
    def __init__(self, args, model, train_dataloader, eval_dataloader):
        self.args = args
        self.model = model
        self.device = args.device
        self.model.to(self.device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.file_name = args.ROOT_DIR + args.data_name + args.model_name + ".pt"

        self.train_hist = pd.DataFrame(columns=["epoch", "loss", "acc"])
        self.val_hist = pd.DataFrame(columns=["epoch", "loss", "acc"])
        self.tail_hist = pd.DataFrame(columns=["epoch", "loss_D", "loss_G"])

        self.lr_dict = {}
        self.optim_dict = {}
        betas = (self.args.adam_beta1, self.args.adam_beta2)

        self.optim_dict["ER"] = Adam(
            list(self.model.embedder.parameters())
            + list(self.model.recovery.parameters()),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.optim_dict["S"] = Adam(
            self.model.supervisor.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.optim_dict["G"] = Adam(
            self.model.generator.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.optim_dict["D"] = Adam(
            self.model.discriminator.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.optim_dict["SR"] = Adam(
            list(self.model.supervisor.parameters())
            + list(self.model.recovery.parameters()),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.lr_dict["ER"] = ReduceLROnPlateau(
            self.optim_dict["ER"], mode="min", factor=0.2, patience=10
        )
        self.lr_dict["S"] = ReduceLROnPlateau(
            self.optim_dict["S"], mode="min", factor=0.2, patience=10
        )
        self.lr_dict["G"] = ReduceLROnPlateau(
            self.optim_dict["G"], mode="min", factor=0.2, patience=10
        )
        self.lr_dict["D"] = ReduceLROnPlateau(
            self.optim_dict["D"], mode="min", factor=0.2, patience=10
        )
        self.lr_dict["SR"] = ReduceLROnPlateau(
            self.optim_dict["SR"], mode="min", factor=0.2, patience=10
        )

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch, stage):
        self.iteration(epoch, stage, self.train_dataloader)

    def valid(self, epoch, stage):
        return self.iteration(epoch, stage, self.eval_dataloader, mode="valid")

    def iteration(self, epoch, stage, dataloader, mode="train"):
        raise NotImplementedError

    def save(self, stage):
        file_name = (
            self.args.ROOT_DIR
            + self.args.data_name
            + self.args.model_name
            + stage
            + ".pt"
        )
        torch.save(self.model.state_dict(), file_name)

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
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x
        )
        self.val_hist = self.val_hist.applymap(
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x
        )

        loss_df = pd.concat(
            [self.train_hist["epoch"], self.train_hist["loss"], self.val_hist["loss"]],
            axis=1,
        )
        loss_df.set_index(["epoch"], inplace=True)
        plot_hist(loss_df.index, loss_df, title)

        # acc_df = pd.concat([self.train_hist['epoch'], self.train_hist['acc'], self.val_hist['acc']], axis=1)
        # acc_df.set_index(['epoch'], inplace=True)
        # plot_hist(acc_df.index, acc_df, 'R2')

        self.train_hist = pd.DataFrame(columns=["epoch", "loss", "acc"])
        self.val_hist = pd.DataFrame(columns=["epoch", "loss", "acc"])

    def evaluate_tail(self, title):
        self.tail_hist = self.tail_hist.applymap(
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x
        )

        tail_df = pd.concat(
            [
                self.tail_hist["epoch"],
                self.tail_hist["loss_D"],
                self.tail_hist["loss_G"],
            ],
            axis=1,
        )
        tail_df.set_index(["epoch"], inplace=True)
        plot_tail(tail_df.index, tail_df, title)

        self.tail_hist = pd.DataFrame(columns=["epoch", "loss_D", "loss_G"])


class FinetuneTrainer(Trainer):
    def __init__(self, args, model, train_dataloader, eval_dataloader):
        super(FinetuneTrainer, self).__init__(
            args, model, train_dataloader, eval_dataloader
        )

    def iteration(self, epoch, stage, dataloader, mode="train"):

        if self.args.verbose:
            # Setting the tqdm progress bar
            rec_data_iter = tqdm.tqdm(
                enumerate(dataloader),
                desc=f"{self.args.model_name} EP_{mode}:{epoch}",
                total=len(dataloader),
                bar_format="{l_bar}{r_bar}",
            )

        else:
            rec_data_iter = enumerate(dataloader)

        if mode == "train":
            self.model.train()
            cur_loss = 0.0
            total_loss = 0.0
            total_acc = 0.0
            total = 0.0
            tail_loss_D = 0.0
            tail_loss_G = 0.0

            for i, batch in rec_data_iter:
                input = batch.to(self.device)
                if stage == "Pretrain_1":
                    x_tilde, loss = self.model(stage, input)
                    self.optim_dict["ER"].zero_grad()
                    loss.backward()
                    self.optim_dict["ER"].step()

                elif stage == "Pretrain_2":
                    x_tilde, loss = self.model(stage, input)
                    self.optim_dict["S"].zero_grad()
                    loss.backward()
                    self.optim_dict["S"].step()

                else:
                    x_hat, loss_G, loss_D, loss = self.model(stage, input)
                    self.optim_dict["G"].zero_grad()
                    loss_G.backward(retain_graph=True)
                    self.optim_dict["G"].step()
                    tail_loss_G += loss_G

                    self.optim_dict["D"].zero_grad()
                    loss_D.backward(retain_graph=True)
                    self.optim_dict["D"].step()
                    tail_loss_D += loss_D

                    self.optim_dict["SR"].zero_grad()
                    loss.backward()
                    self.optim_dict["SR"].step()

                cur_loss = loss
                total_loss += loss
                total_acc += 0
                total += batch.size(0)

            avg_loss = total_loss / len(dataloader)
            avg_acc = 100 * total_acc / total

            new_data = pd.DataFrame(
                [[epoch, avg_loss, avg_acc]], columns=["epoch", "loss", "acc"]
            )
            self.train_hist = pd.concat(
                [self.train_hist.astype("float32"), new_data.astype("float32")], axis=0
            ).reset_index(drop=True)

            if stage == "Finetune":
                avg_tail_loss_D = tail_loss_D / len(dataloader)
                avg_tail_loss_G = tail_loss_G / len(dataloader)

                new_tail = pd.DataFrame(
                    [[epoch, avg_tail_loss_D, avg_tail_loss_G]],
                    columns=["epoch", "loss_D", "loss_G"],
                )
                self.tail_hist = pd.concat(
                    [self.tail_hist.astype("float32"), new_tail.astype("float32")],
                    axis=0,
                ).reset_index(drop=True)

            post_fix = {
                "epoch": epoch,
                "avg_loss": "{:.4f}".format(total_loss / len(dataloader)),
                "cur_loss": "{:.4f}".format(cur_loss),
            }

            if self.args.log_freq > 0 and (epoch + 1) % self.args.log_freq == 0:
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
                        x_hat, loss_G, loss_D, loss = self.model(stage, input)

                    total_loss += loss
                    total_acc += 0
                    total += batch.size(0)

            avg_loss = total_loss / len(dataloader)
            avg_acc = 100 * total_acc / total

            new_data = pd.DataFrame(
                [[epoch, avg_loss, avg_acc]], columns=["epoch", "loss", "acc"]
            )
            self.val_hist = pd.concat(
                [self.val_hist.astype("float32"), new_data.astype("float32")], axis=0
            ).reset_index(drop=True)

            if stage == "Pretrain_1":
                self.lr_dict["ER"].step(avg_loss)

            elif stage == "Pretrain_2":
                self.lr_dict["S"].step(avg_loss)

            else:
                self.lr_dict["G"].step(avg_loss)
                self.lr_dict["D"].step(avg_loss)
                self.lr_dict["SR"].step(avg_loss)

            return avg_loss.cpu().detach().numpy()

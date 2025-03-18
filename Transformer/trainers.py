import tqdm
import torch
import pandas as pd
from utils import plot_hist
from metrics import F1Score
from torch.optim import Adam
from metrics import ensure_tensor_array
from torch.optim.lr_scheduler import ReduceLROnPlateau


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

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.submission_dataloader = submission_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        self.lr_scheduler = ReduceLROnPlateau(self.optim, mode='min', factor=0.2, patience=10)

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
        output = torch.sigmoid(torch.tensor(output))
        return F1Score(output, gt)

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

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, gt = batch

                # Binary cross_entropy
                output, loss = self.model(input_ids, attention_mask, gt)
                accuracy = self.metric(output, gt)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss
                cur_loss = loss
                total_acc += accuracy

            avg_loss = total_loss / len(rec_data_iter)
            avg_acc = total_acc / len(rec_data_iter)

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

            batch_results = []
            with torch.no_grad():
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    if mode != "submission":
                        input_ids, attention_mask, gt = batch
                        output, loss = self.model(input_ids, attention_mask, gt)
                        accuracy = self.metric(output, gt)
                        total_loss += loss
                        total_acc += accuracy

                    else:
                        input = batch
                        output, h_t = self.model(input[0].unsqueeze(1))
                        output = ensure_tensor_array(output)
                        batch_results.extend(output)

            avg_loss = total_loss / len(rec_data_iter)
            avg_acc = total_acc / len(rec_data_iter)

            new_data = pd.DataFrame([[epoch, avg_loss, avg_acc]], columns=["epoch", "loss", "acc"])
            self.val_hist = pd.concat([self.val_hist.astype("float32"), new_data.astype("float32")],
                                      axis=0).reset_index(drop=True)

            if mode != "submission":
                self.lr_scheduler.step(avg_loss)
                return avg_loss.cpu().detach().numpy()

            else:
                pred = pd.DataFrame(batch_results, columns=['y_pred'])
                return pred

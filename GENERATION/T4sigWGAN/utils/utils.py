import os
import random
import pickle

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcdefaults()


def sample_indices(dataset_size, batch_size, device):
    '''
    Use np.random.choice to sample data: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
    '''
    indices = torch.from_numpy(np.random.choice(dataset_size, size=batch_size, replace=False))
    if device == 'cuda':
        indices = indices.cuda()
    else:
        indices = indices

    return indices.long()


def pickle_it(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when the performance is better."""
        if self.verbose:
            print(f"Better performance. Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def save_args_txt(args, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")


def load_data(file_path):
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)

    train = loaded_data["train"]
    val = loaded_data["val"]
    test = loaded_data["test"]

    return train, val, test


def retrainer(trainer, args):
    if args.backbone_weight_path:
        val_loss1 = trainer.valid(0)
        trainer.load(args.backbone_weight_path)
        val_loss2 = trainer.valid(0)

        if val_loss1 > val_loss2:
            print('backbone')
            best_loss = val_loss2
        else:
            print('retrain')
            best_loss = val_loss1
            trainer.load(args.checkpoint_path)
    else:
        val_loss1 = trainer.valid(0)
        best_loss = val_loss1

    return best_loss


def plot_hist(index, hist_df, title):
    plt.figure(figsize=(15, 3), dpi=400)
    plt.plot(index, hist_df.iloc[:, 0], label="train")
    plt.plot(index, hist_df.iloc[:, 1], label="val")
    plt.title(f"Train-Val {title}")
    plt.xlabel("Training Epochs")
    plt.ylabel(f"{title}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"output/{title}_plot.png", bbox_inches='tight')

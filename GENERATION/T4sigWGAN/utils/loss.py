import torch
import torch.nn as nn

def mse(x, y):
    return torch.mean((x - y) ** 2)


def mae(x, y):
    return torch.abs((x - y))

def sigcwgan_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    return torch.norm(sig_pred - sig_fake_conditional_expectation, p=2, dim=1).mean()

def G1(v):
    return v


def G2(e, scale=1):
    return scale * torch.exp(e / scale)


def G2in(e, scale=1):
    return scale ** 2 * torch.exp(e / scale)


def G1_quant(v, W=10.0):
    return - W * v ** 2 / 2


def G2_quant(e, alpha):
    return alpha * e


def G2in_quant(e, alpha):
    return alpha * e ** 2 / 2


# general score function
def S_stats(v, e, X, alpha):
    """
    For a given quantile, here named alpha, calculate the score function value
    """
    if alpha < 0.5:
        rt = ((X<=v).float() - alpha) * (G1(v) - G1(X)) + 1. / alpha * G2(e) * (X<=v).float() * (v - X) + G2(e) * (e - v) - G2in(e)
    else:
        alpha_inverse = 1 - alpha
        rt = ((X>=v).float() - alpha_inverse) * (G1(X) - G1(v)) + 1. / alpha_inverse * G2(-e) * (X>=v).float() * (X - v) + G2(-e) * (v - e) - G2in(-e)
    return torch.mean(rt)


# a specific score function requiring some constraints on VAR and ES, but having better optimization properties
def S_quant(v, e, X, alpha, W=10.0):
    """
    For a given quantile, here named alpha, calculate the score function value
    """
    if alpha < 0.5:
        rt = ((X<=v).float() - alpha) * (G1_quant(v,W) - G1_quant(X,W)) + 1. / alpha * G2_quant(e,alpha) * (X<=v).float() * (v - X) + G2_quant(e,alpha) * (e - v) - G2in_quant(e,alpha)
    else:
        alpha_inverse = 1 - alpha
        rt = ((X>=v).float() - alpha_inverse) * (G1_quant(v,W) - G1_quant(X,W)) + 1. / alpha_inverse * G2_quant(-e,alpha_inverse) * (X>=v).float() * (X - v) + G2_quant(-e,alpha_inverse) * (v - e) - G2in_quant(-e,alpha_inverse)
    return torch.mean(rt)


class Score(nn.Module):
    def __init__(self):
        super(Score, self).__init__()
        self.alphas = [0.05]
        self.score_name = 'quant'
        if self.score_name == 'quant':
            self.score_alpha = S_quant
        elif self.score_name == 'stats':
            self.score_alpha = S_stats
        else:
            self.score_alpha = None

    def forward(self, PNL_validity, PNL):
        # Score
        loss = 0
        for i, alpha in enumerate(self.alphas):
            PNL_var = PNL_validity[:, [2 * i]]
            PNL_es = PNL_validity[:, [2 * i + 1]]
            loss += self.score_alpha(PNL_var, PNL_es, PNL.T, alpha)

        return loss
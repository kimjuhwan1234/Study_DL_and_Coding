from ..utils.Transform import *
from ..utils.gen_thresholds import gen_thresholds
from torch.autograd import Variable
np.random.seed(1)
torch.manual_seed(1)


def Compute_PNL(R, config):
    prices_l = Inc2Price(R)
    port_prices_l = StaticPort(prices_l, 50, config['static_way'], insample=True)
    PNL_BH = BuyHold(prices_l, config['Cap'])
    PNL_l = [PNL_BH]
    for strategy in config['strategies']:
        if strategy == 'Port':
            PNL_BHPort = BuyHold(port_prices_l, config['Cap'])
            PNL_l.append(PNL_BHPort)
        elif strategy == 'MR':
            for percentile_l in config['thresholds_pct']:
                thresholds_array = gen_thresholds(config['data_name'],
                                                  config['tickers'], strategy,
                                                  percentile_l, config['WH'])
                PNL_MR = MeanRev(prices_l, config['Cap'], config['WH'], LR=config['ratios'][0], SR=config['ratios'][1],
                                 ST=thresholds_array[:, -1], LT=thresholds_array[:, -2])
                PNL_l.append(PNL_MR)
        elif strategy == 'TF':
            for percentile_l in config['thresholds_pct']:
                thresholds_array = gen_thresholds(config['data_name'],
                                                  config['tickers'], strategy,
                                                  percentile_l, config['WH'])
                PNL_TF = TrendFollow(prices_l, config['Cap'], config['WH'], LR=config['ratios'][0],
                                     SR=config['ratios'][1],
                                     ST=thresholds_array[:, 0], LT=thresholds_array[:, 1])
                PNL_l.append(PNL_TF)
    PNL = torch.cat(PNL_l, dim=1)
    return PNL


def deterministic_NeuralSort(s, tau):
    n = s.size()[1]
    one = torch.ones((n, 1)).type_as(s)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    B = torch.matmul(A_s, torch.matmul(one, one.t()))
    scaling = (n + 1 - 2 * (torch.arange(n) + 1)).type_as(s)
    C = torch.matmul(s, scaling.unsqueeze(0))
    P_max = (C - B).permute(0, 2, 1)
    sm = torch.nn.Softmax(dim=-1)
    P_hat = sm(P_max / tau)
    return P_hat


class tailGANDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W = config['W']
        self.project = config['project']
        self.alphas = config['alphas']
        self.temp = config['temp']
        self.batch_size = config['batch_size']
        self.model = nn.Sequential(
            nn.Linear(self.batch_size, 256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(128, 2 * len(self.alphas))
        )

    def project_op(self, validity):
        validity_out = validity.clone()  # 새로운 텐서로 복사

        for i, alpha in enumerate(self.alphas):
            v = validity[:, 2 * i].clone()
            e = validity[:, 2 * i + 1].clone()
            indicator = torch.sign(torch.tensor(0.5 - alpha))

            cond = (self.W * v < e).float()
            not_cond = 1.0 - cond

            v_new = indicator * (cond * v + not_cond * (v + self.W * e) / (1 + self.W ** 2))
            e_new = indicator * (cond * e + not_cond * self.W * (v + self.W * e) / (1 + self.W ** 2))

            validity_out[:, 2 * i] = v_new
            validity_out[:, 2 * i + 1] = e_new

        return validity_out

    def forward(self, R):
        R=Variable(R.type(Tensor), requires_grad=True)
        PNL = Compute_PNL(R, self.config)
        PNL_transpose = PNL.T
        PNL_s = PNL_transpose.unsqueeze(-1)
        perm_matrix = deterministic_NeuralSort(PNL_s, self.temp)
        PNL_sort = torch.bmm(perm_matrix, PNL_s)
        PNL_validity = self.model(PNL_sort.reshape(*PNL_transpose.shape))
        if self.project:
            PNL_validity = self.project_op(PNL_validity)
        return PNL, PNL_validity

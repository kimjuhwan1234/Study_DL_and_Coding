import torch
import argparse
from .utils.augmentations import parse_augmentations

parser = argparse.ArgumentParser()

# initial setting
parser.add_argument("--epochs", type=int, default=2, help="number of epochs")
parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# log location
parser.add_argument("--ROOT_DIR", default="result/", type=str)
parser.add_argument("--data_name", default="Stock_", type=str)
parser.add_argument("--model_name", default="T4sigWGAN", type=str)

# trainer parameter
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
parser.add_argument("--log_freq", type=int, default=5, help="per epoch print res")

args, unknown = parser.parse_known_args()

## decoder parameters
decoder_config = {
    'ts_shape': [60, 256],
    'num_heads': 2,
    'k_size': 4,
    'dilations': [1, 4],
    'dropout': 0.2,
}

class Configs:
    def __init__(self, ):
        self.enc_in = 256
        self.dims = [256, 128, 256]
        self.large_size = [5, 5, 3]
        self.small_size = [5, 3, 3]
        self.small_kernel_merged = False
        self.dropout = 0.1
        self.head_dropout = 0.2
        self.revin = True
        self.affine = True
        self.decomposition = True
        self.kernel_size = 25


## supervisor parameters
supervisor_config = Configs()


## generator parameters
encoder_config = {
    "input_dim": 10,
    "hidden_dim": 256,
    "augmentations": [
        {"name": "LeadLag"},
    ],
    "depth": 2,
    "output_dim": 10,
    "len_interval_u": 50,
    "init_fixed": True,
    "batch_size": 128,
    "window_size": 60,
    "device": "cuda:0",
}

if encoder_config.get('augmentations') is not None:
    encoder_config['augmentations'] = parse_augmentations(encoder_config.get('augmentations'))


## generator parameters
logsig_config = {
    "input_dim": 10,
    "hidden_dim": 256,
    "augmentations": [
        {"name": "LeadLag"},
    ],
    "depth": 2,
    "output_dim": 10,
    "len_noise": 1000,
    "len_interval_u": 50,
    "init_fixed": True,
    "batch_size": 128,
    "window_size": 60,
    "device": "cuda:0",
}

if logsig_config.get('augmentations') is not None:
    logsig_config['augmentations'] = parse_augmentations(logsig_config.get('augmentations'))


## discriminator parameters
discriminator_config = {
    'n_epochs': 3000,
    'batch_size': 128,
    'lr_D': 1e-7,
    'lr_G': 1e-6,
    'temp': 0.01,
    'b1': 0.5,
    'b2': 0.999,
    'latent_dim': 1000,
    'n_len': 50000,
    'n_rows': 5,
    'n_cols': 100,
    'n_critic_G': 1,
    'n_critic_D': 1,
    'static_way': 'LShort',
    'strategies': ['Port', 'MR', 'TF'],
    'n_trans': 50,
    'Cap': 10,
    'WH': 30,
    'ratios': [1.0, 1.0],
    'thresholds_pct': [[31, 69]],
    'data_name': 'AAPL_DIS_XOM_INTC_MSFT_AMZN_NVDA_CRM_GOOG_TSLA',
    'tickers': ['AAPL', 'DIS', 'XOM', 'INTC', 'MSFT', 'AMZN', 'NVDA', 'CRM', 'GOOG', 'TSLA'],
    'noise_name': 't5',
    'alphas': [0.05],
    'W': 10.0,
    'score': 'quant',
    'numNN': 10,
    'project': True,
    'version': 'Test1'
}
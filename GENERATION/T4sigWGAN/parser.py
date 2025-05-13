import argparse

import torch

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
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight_decay of adam")
parser.add_argument("--log_freq", type=int, default=499, help="per epoch print res")
parser.add_argument("--verbose", default=False, help="log")

# common model parameter
parser.add_argument("--input_size", type=int, default=10, help="input size of model")
parser.add_argument("--hidden_size", type=int, default=256, help="hidden size of model")
parser.add_argument("--window_size", type=int, default=60, help="number of window_size")
parser.add_argument("--batch_size", type=int, default=128, help="number of batch_size")

args, unknown = parser.parse_known_args()

## decoder parameters
decoder_config = {
    'ts_shape': [args.window_size, args.hidden_size],
    'num_heads': 2,
    'k_size': 5,
    'dilations': [1, 4],
    'dropout': 0.2,
}

## supervisor parameters
supervisor_config = {
    "enc_in": args.hidden_size,
    "dims": [args.hidden_size, 128, args.hidden_size],
    "large_size": [5, 5, 3],
    "small_size": [5, 3, 3],
    "small_kernel_merged": False,
    "revin": True,
    "affine": True,
    "decomposition": True,
    "kernel_size": 25,
}

## encoder parameters
encoder_config = {
    "input_dim": args.input_size,
    "augmentations": [
        {"name": "LeadLag"},
    ],
    "depth": 2,
    "hidden_dim": args.hidden_size,
    "batch_size": args.batch_size,
    "window_size": args.window_size,
    "device": "cuda:0",
    "len_interval_u": 50,
    "init_fixed": True,
}

if encoder_config.get('augmentations') is not None:
    encoder_config['augmentations'] = parse_augmentations(encoder_config.get('augmentations'))

## generator parameters
logsig_config = {
    "input_dim": args.input_size,
    "augmentations": [
        {"name": "LeadLag"},
    ],
    "depth": 2,
    "window_size": args.window_size,
    "hidden_dim": args.hidden_size,
    "device": "cuda:0",
    "len_noise": 1000,
    "len_interval_u": 50,
    "init_fixed": True,
}

if logsig_config.get('augmentations') is not None:
    logsig_config['augmentations'] = parse_augmentations(logsig_config.get('augmentations'))

## discriminator parameters
discriminator_config = {
    'W': 10.0,
    'project': True,
    'alphas': [0.05],
    'temp': 0.01,
    'batch_size': args.batch_size,
    'static_way': 'LShort',
    'Cap': 10,
    'strategies': ['Port', 'MR', 'TF'],
    'thresholds_pct': [[31, 69]],
    'data_name': 'AAPL_DIS_XOM_INTC_MSFT_AMZN_NVDA_CRM_GOOG_TSLA',
    'tickers': ['AAPL', 'DIS', 'XOM', 'INTC', 'MSFT', 'AMZN', 'NVDA', 'CRM', 'GOOG', 'TSLA'],
    'WH': 30,
    'ratios': [1.0, 1.0],
}

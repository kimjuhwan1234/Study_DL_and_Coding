#%%
import gc
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import ALGOS
from os import path as pt
from lib.utils import pickle_it
from lib.algos.base import BaseConfig
from lib.data import get_data, get_data2
from hyperparameters import SIGCWGAN_CONFIGS
from lib.plot import savefig, create_summary

gc.collect()
#%%
def get_algo_config(dataset, data_params):
    """ Get the algorithms parameters. """
    key = dataset
    if dataset == 'VAR':
        key += str(data_params['dim'])
    elif dataset == 'STOCKS':
        key += '_' + '_'.join(data_params['assets'])
    return SIGCWGAN_CONFIGS[key]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_algo(algo_id, base_config, dataset, data_params, x_real):
    if algo_id == 'SigCWGAN':
        algo_config = get_algo_config(dataset, data_params)
        algo = ALGOS[algo_id](x_real=x_real, config=algo_config, base_config=base_config)
    else:
        algo = ALGOS[algo_id](x_real=x_real, base_config=base_config)
    return algo


def run(algo_id, base_config, dataset, data_params={}):
    """ Create the experiment directory, calibrate algorithm, store relevant parameters. """
    experiment_directory = f'./numerical_results/{dataset}/stock/seed=0/SigCWGAN'

    df_2 = get_dataset_configuration()
    x_real = get_data2(df_2.values.reshape(1, -1, 10), base_config.p, base_config.q)
    x_real = x_real.to(base_config.device)
    ind_train = int(x_real.shape[0] * 0.8)
    x_real_train, x_real_test = x_real[:ind_train], x_real[ind_train:]  #train_test_split(x_real, train_size = 0.8)

    algo = get_algo(algo_id, base_config, dataset, data_params, x_real_train)
    # Train the algorithm
    algo.fit()
    # create summary
    create_summary(dataset, base_config.device, algo.G, base_config.p, base_config.q, x_real_test)
    savefig('summary.png', experiment_directory)
    x_fake = create_summary(dataset, base_config.device, algo.G, base_config.p, 8000, x_real_test, one=True)
    print(x_fake.shape)
    savefig('summary_long.png', experiment_directory)
    plt.plot(x_fake.cpu().numpy()[0, :2000])
    savefig('long_path.png', experiment_directory)
    # Pickle generator weights, real path and hyperparameters.
    pickle_it(x_real, pt.join(pt.dirname(experiment_directory), 'x_real.torch'))
    pickle_it(x_real_test, pt.join(pt.dirname(experiment_directory), 'x_real_test.torch'))
    pickle_it(x_real_train, pt.join(pt.dirname(experiment_directory), 'x_real_train.torch'))
    pickle_it(algo.training_loss, pt.join(experiment_directory, 'training_loss.pkl'))
    pickle_it(algo.G.to('cpu').state_dict(), pt.join(experiment_directory, 'G_weights.torch'))
    # Log some results at the end of training
    algo.plot_losses()
    savefig('losses.png', experiment_directory)


def get_dataset_configuration():
    price_df = pd.read_csv('./data/sp500.csv')
    price_df.set_index('datadate', inplace=True)
    df_0 = price_df[['AAPL', 'DIS', 'XOM', 'INTC', 'MSFT', 'AMZN', 'NVDA', 'CRM', 'GOOGL', 'TSLA']]
    df_1 = df_0.dropna(axis=0)
    df_2 = df_1.pct_change().dropna()
    return df_2
#%%
parser = argparse.ArgumentParser()
# Meta parameters
parser.add_argument('-base_dir', default='./numerical_results', type=str)
parser.add_argument('-use_cuda', default=1, action='store_true')
parser.add_argument('-device', default=0, type=int)
parser.add_argument('-num_seeds', default=1, type=int)
parser.add_argument('-initial_seed', default=0, type=int)
#parser.add_argument('-datasets', default=['ARCH', 'STOCKS', 'ECG', 'VAR', ], nargs="+")
parser.add_argument('-datasets', default=['STOCKS', 'ARCH', 'VAR', ], nargs="+")
parser.add_argument('-algos', default=['SigCWGAN', 'GMMN', 'RCGAN', 'TimeGAN', 'RCWGAN', 'CWGAN', ], nargs="+")

# Algo hyperparameters
parser.add_argument('-batch_size', default=200, type=int)
parser.add_argument('-p', default=3, type=int)
parser.add_argument('-q', default=3, type=int)
parser.add_argument('-hidden_dims', default=5 * (64,), type=tuple)
parser.add_argument('-total_steps', default=1000, type=int)

args, unknown = parser.parse_known_args()

set_seed(0)

base_config = BaseConfig(
    device='cuda:{}'.format(args.device) if args.use_cuda and torch.cuda.is_available() else 'cpu',
    batch_size=args.batch_size,
    hidden_dims=args.hidden_dims,
    seed=0,
    p=args.p,
    q=args.q,
    total_steps=args.total_steps,
    mc_samples=1000,
)

data_params = {'dim': 10}
base_config.device
#%%
run(
    algo_id='SigCWGAN',
    base_config=base_config,
    data_params=data_params,
    dataset='VAR',
)
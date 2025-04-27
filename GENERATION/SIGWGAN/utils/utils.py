import json
import pickle
from datetime import datetime, timedelta
import tqdm

import torch
import numpy as np
import random
from .augmentations import augment_path_and_compute_signatures
from sklearn.linear_model import LinearRegression

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

def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()

def set_seed(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def regression_on_linear_functional(x_future, x_past, sig_config):
    sig_future = augment_path_and_compute_signatures(x_future,sig_config["depth"])
    sig_past = augment_path_and_compute_signatures(x_past,sig_config["depth"])
    linear_functional = LinearRegression()
    linear_functional.fit(to_numpy(sig_past),to_numpy(sig_future))
    return linear_functional
    
def predict(linear_functional, sig_past: torch.Tensor):
    return torch.from_numpy(linear_functional.predict(sig_past)).float().to(sig_past.device)

def load_obj(filepath):
    """ Generic function to load an object. """
    if filepath.endswith('pkl'):
        loader = pickle.load
    elif filepath.endswith('pt'):
        loader = torch.load
    elif filepath.endswith('json'):
        import json
        loader = json.load
    else:
        raise NotImplementedError()
    with open(filepath, 'rb') as f:
        return loader(f)

def save_obj(obj: object, filepath: str):
    """ Generic function to save an object with different methods. """

    if filepath.endswith('pkl'):
        saver = pickle.dump
    elif filepath.endswith('pt'):
        saver = torch.save
    elif filepath.endswith('json'):
        with open(filepath, 'w') as f:
            json.dump(obj, f, indent=4)
        return 0
    else:
        raise NotImplementedError()
    with open(filepath, 'wb') as f:
        saver(obj, f)
    return 0

# Probably no need    
def rolling_period_resample(dataset, period, n_lags):
    period_dict = {
        'd': 0,
        'h': 0
    }
    assert period[-1] in period_dict.keys()
    period_dict[period[-1]] = int(period[:-1])
    time_period = timedelta(days=period_dict['d'], hours=period_dict['h'])
    time_clip = time_period / n_lags
    dataset_length = len(dataset)

    # print('length', dataset_length)

    def find_data(start_i, target_t):
        i_point = 0
        kk = 0
        while start_i + i_point < dataset_length - 1:
            if dataset[start_i + i_point][0] <= target_t <= dataset[start_i + i_point + 1][0]:
                return start_i + i_point
            else:
                if dataset[start_i + i_point][0] > target_t:
                    print(f'find {target_t} from index {start_i}={dataset[start_i]}. '
                          f'current {dataset[start_i + i_point][0]}')
                    kk += 1
                    if kk > 5:
                        exit()
                i_point += 1

    rolled_dataset = []
    rolled_data_idx = [0] * (n_lags - 1)
    for i, data in enumerate(tqdm(dataset)):
        start_time = datetime.fromtimestamp(int(data[0]))
        end_time = (start_time + time_period).timestamp()
        if dataset[-1][0] < end_time:
            break

        rolled_data = [data]
        time_point = (start_time + time_clip).timestamp()

        data_idx = 0
        for k, idx in enumerate(rolled_data_idx):
            idx = max(idx, data_idx)
            assert dataset[idx][0] < time_point
            data_idx = find_data(idx, time_point)
            rolled_data.append(dataset[data_idx])
            rolled_data_idx[k] = data_idx
            time_point = (start_time + time_clip * len(rolled_data)).timestamp()

        # print(rolled_data_idx)
        rolled_dataset.append(np.stack(rolled_data))

    return np.stack(rolled_dataset)
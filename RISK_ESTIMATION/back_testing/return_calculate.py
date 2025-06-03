import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def return_Table(weight_df: pd.DataFrame, momentum_df: pd.DataFrame, result_df: pd.DataFrame,
                 log: bool, type: str, stop_loss: bool, fee: float) -> pd.DataFrame:
    '''
    weight_df (pd.DataFrame): float weight
    momentum_df (pd.DataFrame): simple return
    type (str): Neutral, Long, Short
    stop_loss (bool): if loss is lower than 30%, then investing must be stopped.
    fee (float): transaction cost per trade (given as simple return)
    log (bool): convert simple return to log return
    return: updated result_df (pd.DataFrame)
    '''

    if log:
        momentum_df = momentum_df.applymap(lambda x: np.log1p(x))

    if len(weight_df.columns) != len(momentum_df.columns):
        momentum_df['CASH']=np.log1p(0.000119)

    if type == 'Neutral':
        return_df = weight_df * momentum_df

    elif type == 'Long':
        weight_df = weight_df.where(weight_df > 0, 0)
        return_df = weight_df * momentum_df

    else:  # Short
        weight_df = weight_df.where(weight_df < 0, 0)
        return_df = weight_df * momentum_df

    if stop_loss:
        return_df = return_df.applymap(lambda x: max(x, np.log(0.7)))  # -30% stop loss → log(0.7)

    mask = (weight_df != 0) & momentum_df.isna()
    return_df[mask] = np.log(0.5)  # -50% 손실 → log(0.5)
    return_df = return_df.sum(axis=1)

    if fee:
        fee_log = np.log(1 - fee)
        non_zero_count = weight_df.astype(bool).sum(axis=1)
        return_df = return_df + non_zero_count * fee_log

    return_df = return_df.dropna()
    result_df = pd.concat([result_df, return_df], axis=1).sort_index()
    return result_df


def cal_turnover(weight_df: pd.DataFrame)->float:
    turnover = (weight_df.diff().abs().sum(axis=1)).sum() / 2
    return f"{turnover:.2f}"


def plot_result(result_df, apply_log: bool):
    result_df = result_df.cumsum() if apply_log else result_df.cumprod()
    result_df = result_df.dropna(axis=0)

    color_dict = {
        'Cointegration': 'darkgrey',  # Darker shade of grey
        'Reversal': 'lightgrey',  # Lighter shade of grey
        'FTSE 100': 'grey',  # Standard grey
        'S&P 500': 'black',
        'GAT_TCN': 'brown',
        'LSTM': 'yellow',
    }

    plt.figure(figsize=(12, 5), dpi=400)
    import matplotlib.cm as cm

    handles = []
    for i, key in enumerate(result_df.columns):
        cmap = cm.get_cmap('tab10', len(result_df.columns))  # keys: 라인 수
        color = cmap(i)  # cmap으로 색상 추출
        if key in color_dict.keys():
            if key =='LSTM' or key == 'GAT_TCN':
                line, = plt.plot(result_df.index, result_df.loc[:, key].fillna(method='ffill'),
                                 label=key, color=color_dict[key])
            else:
                line, = plt.plot(result_df.index, result_df.loc[:, key].fillna(method='ffill'),
                                 label=key, color=color_dict[key], linestyle='--')
        else:
            line, = plt.plot(result_df.index, result_df.loc[:, key].fillna(method='ffill'),
                             label=key, color=color)
        handles.append(line)

    plt.title(f'Backtesting Results')
    plt.xlabel('Date')
    if apply_log:
        plt.ylabel(f'Cumulative Log-returns')
    else:
        plt.ylabel(f'Cumulative Simple-returns')

    plt.xticks(rotation=45)
    plt.legend(handles=handles, loc='upper left')
    plt.tight_layout()
    plt.show()

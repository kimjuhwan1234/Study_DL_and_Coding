import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

tickers = ["^GSPC"]
global_data = None


warnings.filterwarnings("ignore")


def return_Table(weight_df: pd.DataFrame, price_df: pd.DataFrame, result_df: pd.DataFrame,
                 rebalance_freq: str = 'D', fee: float = 0.0002, turnover_multiplier: float = 1.0,
                 stop_loss: bool = True, type: str = 'Neutral'
                 ) -> pd.DataFrame:
    '''
    weight_df (pd.DataFrame): float weight
    price_df (pd.DataFrame): price data for calculating returns
    type (str): Neutral, Long, Short
    stop_loss (bool): if loss is lower than 30%, then investing must be stopped.
    fee (float): base transaction cost per trade (given as simple return)
    turnover_multiplier (float): multiplier for turnover-based fee calculation
    rebalance_freq (str): rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
    add_sp500_benchmark (bool): whether to add S&P500 benchmark
    log (bool): convert simple return to log return
    return: updated result_df (pd.DataFrame)
    '''

    # price_df가 이미 momentum 데이터인지 확인하고 처리
    # price_df에서 momentum 계산
    
    price_df.index = pd.to_datetime(price_df.index)
    weight_df.index = pd.to_datetime(weight_df.index)
    
    if rebalance_freq == 'A':
        momentum_df = price_df
    else:
        momentum_df = price_df.resample(rebalance_freq).last().dropna(
            axis=0).pct_change().dropna()
    # 리밸런싱 날짜에만 weight를 적용하고, 나머지는 이전 weight 유지
    weight_df_rebalanced = weight_df.reindex(
        momentum_df.index, method='ffill')

    # 3. log 변환
    momentum_df = momentum_df.applymap(lambda x: np.log1p(x))

    # 4. CASH 컬럼 추가 (필요시)
    if len(weight_df_rebalanced.columns) != len(momentum_df.columns):
        momentum_df['CASH'] = np.log1p(0.000119)
        # CASH에 대한 weight 추가
        cash_weight = 1 - weight_df_rebalanced.sum(axis=1)
        weight_df_rebalanced['CASH'] = cash_weight

    # 5. 투자 유형에 따른 weight 조정
    if type == 'Neutral':
        return_df = weight_df_rebalanced * momentum_df
    elif type == 'Long':
        weight_df_rebalanced = weight_df_rebalanced.where(
            weight_df_rebalanced > 0, 0)
        return_df = weight_df_rebalanced * momentum_df
    else:  # Short
        weight_df_rebalanced = weight_df_rebalanced.where(
            weight_df_rebalanced < 0, 0)
        return_df = weight_df_rebalanced * momentum_df

    # 6. Stop loss 적용
    if stop_loss:
        return_df = return_df.applymap(lambda x: max(
            x, np.log(0.7)))  # -30% stop loss

    # 7. NaN 처리
    mask = (weight_df_rebalanced != 0) & momentum_df.isna()
    return_df[mask] = np.log(0.5)  # -50% 손실
    return_series = return_df.sum(axis=1)

    # 8. 수수료 계산
    if fee:
        # Calculate turnover for each period
        turnover_per_period = 0.5 * weight_df_rebalanced.diff().abs().sum(axis=1)

        # Only apply fee when there's actual turnover (trading)
        dynamic_fee = np.where(
            turnover_per_period > 0,  # Only when there's turnover
            fee + (turnover_per_period *
                   turnover_multiplier * fee),  # Dynamic fee
            0  # No fee when no turnover
        )

        # Convert to log return
        fee_log = np.log(1 - dynamic_fee)

        # Apply fee to returns
        return_series = return_series + fee_log

    global global_data
    if 'S&P 500' not in result_df.columns:
        try:
            global_data = yf.download(
                tickers, start=price_df.index[0], end=price_df.index[-1])['Close']
            
            if rebalance_freq == 'A':
                spx_series = global_data['^GSPC'].sort_index()
                idx_sorted = price_df.index.sort_values()
                spx_aligned_sorted = spx_series.asof(idx_sorted)
                spx_aligned = pd.Series(spx_aligned_sorted.values, index=idx_sorted).reindex(price_df.index)
                sp500_returns = spx_aligned.pct_change().dropna()
            else:
                sp500_returns = global_data['^GSPC'].resample(
                    rebalance_freq).last().dropna(axis=0).pct_change().dropna()
            sp500_returns = sp500_returns.apply(lambda x: np.log1p(x))
            sp500_returns.name = 'S&P 500'
            sp500_returns = sp500_returns.reindex(return_series.index)
            return_df = pd.concat([sp500_returns, return_series], axis=1)
        except Exception as e:
            print(e, 'weight_df index should be datetime.')
            return weight_df.index
    else:
        # S&P 500가 이미 있는 경우, return_series만 DataFrame으로 변환
        return_df = return_series.to_frame()

    result_df = pd.concat([result_df, return_df], axis=1).sort_index()
    return result_df


def cal_turnover(weight_df: pd.DataFrame, rebalance_freq: str = "M") -> pd.Series:
    """
    주기별 포트폴리오 turnover 계산
    turnover = 0.5 * Σ |w_t - w_{t-1}|
    
    Parameters
    ----------
    weight_df : pd.DataFrame
        각 자산별 weight (index=날짜, columns=자산)
    rebalance_freq : str
        리밸런스 주기 ('D','W','M','Q' 등)
        
    Returns
    -------
    turnover_per_period : pd.Series
        각 리밸런스 시점의 turnover
    """
    # 리밸런스 주기별로 weight 뽑기
    if rebalance_freq == 'A':
        rebalance_freq = 'D'
    sampled = weight_df.resample(rebalance_freq).last().dropna(how="all")

    # turnover = 0.5 * sum(abs(diff))
    turnover_per_period = 0.5 * sampled.diff().abs().sum(axis=1)
    return turnover_per_period.dropna()


def cal_dynamic_fee(turnover_per_period: pd.Series,
                    fee: float = 0.002,
                    turnover_multiplier: float = 0.5) -> pd.Series:
    """
    turnover 기반 동적 수수료 계산
    
    Parameters
    ----------
    turnover_per_period : pd.Series
        리밸런스 주기별 turnover
    fee : float
        기본 수수료율 (예: 0.001 = 0.1%)
    turnover_multiplier : float
        turnover에 따른 수수료 증가 배수
    
    Returns
    -------
    fee_log : pd.Series
        로그 수익률로 변환된 수수료 효과
    """
    dynamic_fee = np.where(
        turnover_per_period > 0,
        fee + turnover_per_period * turnover_multiplier * fee,
        0
    )
    fee_log = np.log(1 - dynamic_fee)
    return pd.Series(fee_log, index=turnover_per_period.index)


def plot_result(result_df):
    result_df = result_df.cumsum()
    result_df = result_df.dropna(axis=0)

    color_dict = {
        'Cointegration': 'darkgrey',  # Darker shade of grey
        'Reversal': 'lightgrey',  # Lighter shade of grey
        'FTSE 100': 'grey',  # Standard grey
        'S&P 500': 'black',
    }

    plt.figure(figsize=(12, 5), dpi=400)
    import matplotlib.cm as cm

    handles = []
    for i, key in enumerate(result_df.columns):
        cmap = cm.get_cmap('tab10', len(result_df.columns))  # keys: 라인 수
        color = cmap(i)  # cmap으로 색상 추출
        if key in color_dict.keys():
            line, = plt.plot(result_df.index, result_df.loc[:, key].fillna(method='ffill'),
                             label=key, color=color_dict[key], linestyle='--')
        else:
            line, = plt.plot(result_df.index, result_df.loc[:, key].fillna(method='ffill'),
                             label=key, color=color)
        handles.append(line)

    plt.title(f'Backtesting Results')
    plt.xlabel('Date')
    plt.ylabel(f'Cumulative Log-returns')

    plt.xticks(rotation=45)
    plt.legend(handles=handles, loc='upper left')
    plt.tight_layout()
    plt.show()

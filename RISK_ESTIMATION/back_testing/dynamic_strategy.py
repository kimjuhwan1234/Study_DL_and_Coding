import numpy as np
import pandas as pd


def get_momentum(df_0, period):
    df_1 = df_0.dropna(axis=0)
    df_2 = df_1.pct_change(periods=period).dropna()
    return df_2


def dual_momentum_strategy(momentum_12d: pd.DataFrame, top_n: int = 5):
    weight_df = pd.DataFrame(index=momentum_12d.index, columns=momentum_12d.columns, data=0)

    for date in momentum_12d.index:
        # 1. 해당 날짜 모멘텀 데이터 가져오기
        mom = momentum_12d.loc[date].dropna()

        # 2. 모멘텀 상위 top_n 자산 고르기 (절대 모멘텀 > 0 필터링 포함)
        top_assets = mom[mom > 0].sort_values(ascending=False).head(top_n).index.tolist()

        if len(top_assets) == 0:
            # 고를 자산이 없으면 수익률 0
            weight_df.loc[date] = np.zeros(10)
            continue

        # 3. 다음 날짜 수익률 가져오기
        try:
            next_date_idx = momentum_12d.index.get_loc(date) + 1
            next_date = momentum_12d.index[next_date_idx]
        except (KeyError, IndexError):
            # returns에 next_date가 없으면 끝내기
            break

        # 4. 포트폴리오 수익률 (선택된 자산 평균)
        weight_df.loc[next_date, top_assets] = np.ones(len(top_assets)) / len(top_assets)
    return weight_df


def gtaa_aggressive3_strategy(momentum_12d: pd.DataFrame, price_df: pd.DataFrame, Cash: str = 'CASH', top_n: int = 3):
    momentum_12d['CASH'] = 0.0

    assets = [col for col in momentum_12d.columns if col != 'CASH']
    ma_10m = price_df.rolling(window=10).mean()
    weight_df = pd.DataFrame(index=momentum_12d.index, columns=momentum_12d.columns, data=0.0)

    for date in momentum_12d.index:
        try:
            next_date_idx = momentum_12d.index.get_loc(date) + 1
            next_date = momentum_12d.index[next_date_idx]
        except (KeyError, IndexError):
            break  # 마지막 날짜는 처리 안 함

        mom = momentum_12d.loc[date, assets].dropna()
        selected = mom.sort_values(ascending=False).head(top_n)

        valid_assets = []
        for asset in selected.index:
            try:
                if price_df.at[date, asset] > ma_10m.at[date, asset]:
                    valid_assets.append(asset)
            except KeyError:
                continue

        if len(valid_assets) == 0:
            weight_df.loc[next_date, Cash] = 1.0
        else:
            weight_df.loc[next_date, valid_assets] = 1.0 / top_n
            if len(valid_assets) < top_n:
                cash_weight = (top_n - len(valid_assets)) / top_n
                weight_df.loc[next_date, Cash] = cash_weight

    return weight_df


def gtaa5_momentum_strategy(momentum_12d: pd.DataFrame, price_df: pd.DataFrame, Cash: str = 'CASH'):
    momentum_12d['CASH'] = 0.0  # 현금 자산은 고정 가격

    assets = [col for col in momentum_12d.columns if col != 'CASH']
    # 10개월 이동평균
    ma_10m = price_df.rolling(window=10).mean()

    # 결과 저장
    weight_df = pd.DataFrame(index=momentum_12d.index, columns=momentum_12d.columns, data=0.0)

    for date in momentum_12d.index:
        try:
            next_date_idx = momentum_12d.index.get_loc(date) + 1
            next_date = momentum_12d.index[next_date_idx]
        except (KeyError, IndexError):
            break  # 마지막 시점은 제외

        # Step 1: 해당 날짜 모멘텀 기준 상위 5개 선정
        mom = momentum_12d.loc[date].dropna()
        top5_assets = mom.sort_values(ascending=False).head(5).index.tolist()

        # Step 2: 10개월 이동평균 기준 조건 필터링
        valid_assets = []
        for asset in top5_assets:
            try:
                if price_df.loc[date, asset] > ma_10m.loc[date, asset]:
                    valid_assets.append(asset)
            except KeyError:
                continue

        # Step 3: 가중치 배분
        if len(valid_assets) == 0:
            weight_df.loc[next_date, Cash] = 1.0
        else:
            weight = 1.0 / 5  # 항상 5등분
            for asset in valid_assets:
                weight_df.loc[next_date, asset] = weight
            if len(valid_assets) < 5:
                weight_df.loc[next_date, Cash] += (5 - len(valid_assets)) * weight

    return weight_df


def mean_reversion_zscore_strategy(price_df: pd.DataFrame, window: int = 5, top_n: int = 3):
    """
    z-score 기반 mean-reversion 전략.
    - momentum_12d: 모멘텀 (최근 12일 수익률)
    - price_df: 자산별 가격 데이터
    - window: 이동 평균 기간
    - top_n: z-score가 낮은 자산 중 매수할 개수
    - output: weight_df (날짜 x 자산 비중)
    """
    weight_df = pd.DataFrame(index=price_df.index, columns=price_df.columns, data=0.0)

    for t in range(window, len(price_df) - 1):
        next_date = price_df.index[t + 1]

        # 과거 window 길이만큼 평균 및 표준편차
        hist_prices = price_df.iloc[t - window:t]
        mean = hist_prices.mean()
        std = hist_prices.std().replace(0, np.nan)

        # z-score = (현재가 - 평균) / std
        current_price = price_df.iloc[t]
        zscore = (current_price - mean) / std

        # 하위 z-score 자산 선택
        selected = zscore.sort_values().head(top_n).index
        weight_df.loc[next_date, selected] = 1.0 / top_n

    return weight_df


def trend_following_zscore_strategy(price_df: pd.DataFrame, short_window: int = 5,
                                    long_window: int = 10, top_n: int = 3):
    """
    z-score 기반 trend-following 전략.
    - momentum_12d: 모멘텀 (최근 12일 수익률)
    - price_df: 자산별 가격 데이터
    - short_window: 짧은 이동평균 윈도우
    - long_window: 긴 이동평균 윈도우
    - top_n: z-score가 높은 자산 중 매수할 개수
    - output: weight_df (날짜 x 자산 비중)
    """
    weight_df = pd.DataFrame(index=price_df.index, columns=price_df.columns, data=0.0)

    for t in range(long_window, len(price_df) - 1):
        next_date = price_df.index[t + 1]

        # 과거 short, long 평균
        short_ma = price_df.iloc[t - short_window:t].mean()
        long_ma = price_df.iloc[t - long_window:t].mean()

        # z-score = (short_ma - long_ma) / 0.01
        zscore = (short_ma - long_ma) / 0.01

        # 높은 z-score (상승추세) 자산 선택
        selected = zscore.sort_values(ascending=False).head(top_n).index
        weight_df.loc[next_date, selected] = 1.0 / top_n

    return weight_df


def buy_and_hold_strategy(price_df: pd.DataFrame):
    weight_df = pd.DataFrame(index=price_df.index, columns=price_df.columns, data=1 / (len(price_df.columns)))
    return weight_df

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

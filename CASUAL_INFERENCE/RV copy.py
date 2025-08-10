import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
import sys
import os
import pandas as pd
import numpy as np

# 현재 스크립트 파일의 디렉토리를 기준으로 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 작업 디렉토리를 스크립트 위치로 변경

# ETF 가격 데이터 읽기
try:
    close_df_sorted = pd.read_csv(
        'data/ETF_volume.csv', index_col=0, parse_dates=True)
    print(f"데이터 로드 완료: {close_df_sorted.shape}")
    print(f"컬럼: {list(close_df_sorted.columns)}")
except FileNotFoundError:
    print("ETF_volume.csv 파일을 찾을 수 없습니다. 먼저 EDA.ipynb를 실행하여 데이터를 생성해주세요.")
    exit()


def calculate_realized_volatility(prices, periods, window=20):
    """Realized volatility 계산 (rolling window 사용)"""

    log_returns = np.log(prices / prices.shift(periods))

    # rolling window로 realized volatility 계산
    # sqrt(sum of squared returns / window_size)
    realized_vol = log_returns.rolling(window=window).apply(
        lambda x: np.sqrt(np.sum(x**2) / len(x)) * np.sqrt(252),  # 연율화
        raw=True
    )

    return realized_vol


# Realized volatility 계산 (20일 rolling window)
realized_vol_3d = calculate_realized_volatility(close_df_sorted, 3, window=20)
realized_vol_15d = calculate_realized_volatility(
    close_df_sorted, 15, window=20)
realized_vol_1m = calculate_realized_volatility(close_df_sorted, 30, window=20)


# 컬럼명 변경
realized_vol_3d.columns = [col.replace(
    '_volume', '_volume_3d') for col in realized_vol_3d.columns]
realized_vol_15d.columns = [col.replace(
    '_volume', '_volume_15d') for col in realized_vol_15d.columns]
realized_vol_1m.columns = [col.replace(
    '_volume', '_volume_1m') for col in realized_vol_1m.columns]


realized_vol_3d.index = pd.to_datetime(realized_vol_3d.index)
realized_vol_15d.index = pd.to_datetime(realized_vol_15d.index)
realized_vol_1m.index = pd.to_datetime(realized_vol_1m.index)

# realized_vol_3d.to_csv('data/volume_3d.csv', encoding='utf-8')
# realized_vol_15d.to_csv('data/volume_15d.csv', encoding='utf-8')
# realized_vol_1m.to_csv('data/volume_1m.csv', encoding='utf-8')

# # images 폴더 생성 및 모든 종목 쌍의 그래프 저장

# # images 폴더 구조 생성
# vol_types = ['volume_3d', 'volume_15d', 'volume_1m']
# vol_dataframes = [realized_vol_3d, realized_vol_15d, realized_vol_1m]

# # 종목명 추출 (컬럼명에서 _RV_ 부분 제거)
# tickers = [col.replace('_volume_3d', '').replace('_volume_15d', '').replace('_volume_1m', '')
#            for col in realized_vol_3d.columns]
# tickers = list(set(tickers))  # 중복 제거
# print(f"총 종목 수: {len(tickers)}")
# print(f"종목 리스트: {tickers}")

# # 17C2 = 136개의 쌍 생성
# pairs = list(itertools.combinations(tickers, 2))
# print(f"총 쌍의 수: {len(pairs)}")

# # 각 volatility 타입별로 폴더 생성 및 그래프 저장
# for vol_type, vol_df in zip(vol_types, vol_dataframes):
#     print(f"\n{vol_type} 처리 중...")

#     # 폴더 생성
#     vol_folder = f"images/{vol_type}"
#     os.makedirs(vol_folder, exist_ok=True)

#     # 각 쌍에 대해 그래프 생성 및 저장
#     for i, (ticker1, ticker2) in enumerate(pairs):
#         print(f"  진행률: {i+1}/{len(pairs)} - {ticker1} vs {ticker2}")

#         # 그래프 생성
#         plt.figure(figsize=(15, 6), dpi=150)

#         # 두 종목의 volatility 플롯
#         plt.plot(vol_df.index, vol_df[f'{ticker1}_{vol_type}'],
#                  label=ticker1, linewidth=1.5, alpha=0.8)
#         plt.plot(vol_df.index, vol_df[f'{ticker2}_{vol_type}'],
#                  label=ticker2, linewidth=1.5, alpha=0.8)

#         # 그래프 꾸미기
#         plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # 6개월 간격
#         plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#         plt.xticks(rotation=45)
#         plt.xlabel('Date')
#         plt.ylabel(f'{vol_type}')
#         plt.title(f'{vol_type}: {ticker1} vs {ticker2}')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()

#         # 파일명 생성 (특수문자 제거)
#         filename = f"{vol_folder}/{ticker1}_vs_{ticker2}_{vol_type}.png"
#         plt.savefig(filename, dpi=150, bbox_inches='tight')
#         plt.close()  # 메모리 절약을 위해 그래프 닫기

#         # 진행률 표시 (10개마다)
#         if (i + 1) % 10 == 0:
#             print(f"    {i+1}/{len(pairs)} 완료")

# print(f"\n모든 그래프 저장 완료!")
# print(f"저장된 폴더:")
# for vol_type in vol_types:
#     print(f"- images/{vol_type}/ (136개 파일)")

# # 간단한 통계 정보 출력
# print(f"\n=== {vol_type} 통계 요약 ===")
# for vol_df, vol_type in zip(vol_dataframes, vol_types):
#     print(f"\n{vol_type}:")
#     print(f"  데이터 형태: {vol_df.shape}")
#     print(
#         f"  결측값 비율: {(vol_df.isnull().sum().sum() / (vol_df.shape[0] * vol_df.shape[1]) * 100):.2f}%")
#     print(f"  평균 RV: {vol_df.mean().mean():.4f}")
#     print(f"  최대 RV: {vol_df.max().max():.4f}")
#     print(f"  최소 RV: {vol_df.min().min():.4f}")


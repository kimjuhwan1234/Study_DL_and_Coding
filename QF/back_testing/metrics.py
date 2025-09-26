import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy import stats

# 만약 period가 문자열이면 먼저 Timestamp로 변환

pd.set_option('display.precision', 5)


class METRICS:
    def __init__(self, result_df: pd.DataFrame):
        # Convert index to datetime and ensure proper format
        if not isinstance(result_df.index, pd.DatetimeIndex):
            result_df.index = pd.to_datetime(result_df.index)

        self.result_df = result_df

        # Detect frequency automatically
        self.frequency = self._detect_frequency()
        self.periods_per_year = self._get_periods_per_year()

    def _detect_frequency(self):
        """Detect the frequency of the data automatically"""
        if len(self.result_df) < 2:
            return 'D'  # Default to daily

        # Calculate time differences
        time_diffs = self.result_df.index.to_series().diff().dropna()

        # Get the most common time difference
        most_common_diff = time_diffs.mode().iloc[0]

        # Convert to frequency string
        if most_common_diff <= pd.Timedelta(days=1):
            return 'D'  # Daily
        elif most_common_diff <= pd.Timedelta(days=7):
            return 'W'  # Weekly
        elif most_common_diff <= pd.Timedelta(days=31):
            return 'M'  # Monthly
        elif most_common_diff <= pd.Timedelta(days=93):
            return 'Q'  # Quarterly
        else:
            return 'Y'  # Yearly

    def _get_periods_per_year(self):
        """Get the number of periods per year based on frequency"""
        frequency_map = {
            'D': 252,    # Trading days per year
            'W': 52,     # Weeks per year
            'M': 12,     # Months per year
            'Q': 4,      # Quarters per year
            'Y': 1       # Years per year
        }
        # Default to 252 if unknown
        return frequency_map.get(self.frequency, 252)

    def cal_describe(self, period: tuple = None):
        self.annual_statistics = pd.DataFrame()
        self.temp = self.result_df
        if period is not None:
            # Handle both string and datetime inputs
            if isinstance(period[0], str):
                period = tuple(pd.to_datetime(p).date() for p in period)
            self.temp = self.result_df.loc[period[0]:period[1], :]

        for col in (self.temp.columns):
            statistic_dict = {}
            row = self.temp.loc[:, col].dropna()
            statistic_dict['Date'] = (
                f"{row.index[0].strftime('%Y-%m-%d')} ~ {row.index[-1].strftime('%Y-%m-%d')}")
            statistic_dict['count'] = len(row.index)
            statistic_dict['cumulative return'] = (
                np.exp(np.sum(row)) - 1) * 100
            statistic_dict['annualized return mean'] = np.exp(
                np.mean(row) * self.periods_per_year) - 1
            statistic_dict['annualized return std'] = np.exp(
                np.std(row) * np.sqrt(self.periods_per_year)) - 1
            statistic_dict['annualized return downside std'] = (np.exp(np.std(row[row < 0]) * np.sqrt(self.periods_per_year)) - 1) if len(
                row[row < 0]) > 1 else 0.01
            statistic_dict['sharpe ratio'] = (np.exp(np.mean(
                row) * self.periods_per_year) - 1) / statistic_dict['annualized return std']
            statistic_dict['sortino ratio'] = (np.exp(np.mean(row) * self.periods_per_year) - 1) / statistic_dict[
                'annualized return downside std']
            statistic_dict['gross profit'] = row[row > 0].sum()
            statistic_dict['gross loss'] = row[row < 0].sum()
            statistic_dict['profit factor'] = statistic_dict['gross profit'] / \
                np.abs(statistic_dict['gross loss'])
            statistic_dict['maximum drawdown'] = self.cal_MDD(row)
            statistic_dict['calmar ratio'] = self.cal_calmar_ratio(row)
            statistic_dict['profitable years'] = self.cal_profit_loss_years(
                row, True)
            statistic_dict['unprofitable years'] = self.cal_profit_loss_years(
                row, False)

            statistic_df = pd.DataFrame(
                data=statistic_dict.values(), index=statistic_dict.keys(), columns=[col])
            self.annual_statistics = pd.concat(
                [self.annual_statistics, statistic_df], axis=1)

        self.cal_t_statistics()

    def cal_MDD(self, row):
        cumulative_returns = np.exp(np.cumsum(row))
        max_drawdown = 0
        peak = 1  # 초기 최대 누적 수익은 1 (시작값)

        # 각 시점에서 최대 손실 계산
        for k in range(len(cumulative_returns)):
            if cumulative_returns[k] > peak:
                peak = cumulative_returns[k]
            else:
                drawdown = (peak - cumulative_returns[k]) / peak
                max_drawdown = max(max_drawdown, drawdown)

        return -max_drawdown

    def cal_calmar_ratio(self, row):
        cumulative_returns = np.exp(np.cumsum(row))
        max_drawdown = 0
        peak = 1  # 초기 최대 누적 수익은 1 (시작값)

        # 각 시점에서 최대 손실 계산
        for k in range(len(cumulative_returns)):
            if cumulative_returns[k] > peak:
                peak = cumulative_returns[k]
            else:
                drawdown = (peak - cumulative_returns[k]) / peak
                max_drawdown = max(max_drawdown, drawdown)

        calmar = (np.exp(np.mean(row) * self.periods_per_year) - 1) / \
            abs(max_drawdown)
        return calmar

    def cal_profit_loss_years(self, row, profit):
        # Ensure index is datetime
        if not isinstance(row.index, pd.DatetimeIndex):
            row.index = pd.to_datetime(row.index)

        # Group by year and sum returns
        yearly_returns = row.groupby(row.index.year).sum()

        if profit:
            return yearly_returns[yearly_returns > 0].count()
        else:
            return yearly_returns[yearly_returns < 0].count()

    def cal_t_statistics(self):
        t_test = pd.DataFrame(
            index=['t-statistic', 'p_value'], columns=self.annual_statistics.columns)
        for i in range(len(self.annual_statistics.columns)):
            t_statistic, p_value = stats.ttest_ind_from_stats(self.annual_statistics.iloc[3, i],
                                                              self.annual_statistics.iloc[4, i],
                                                              self.annual_statistics.iloc[1, i],
                                                              self.annual_statistics.iloc[3, 0],
                                                              self.annual_statistics.iloc[4, 0],
                                                              self.annual_statistics.iloc[1, 0])
            t_test.iloc[0, i] = t_statistic
            t_test.iloc[1, i] = p_value

        self.annual_statistics = pd.concat(
            [self.annual_statistics, t_test], axis=0)

    def cal_monthly_statistics(self):
        # Rename to be more generic (not just monthly)
        period_stats = pd.DataFrame(
            index=['Mean', 'Standard deviation', 'Standard error', 't-statistic', 'Min', '25%', '50%', '75%', 'Max',
                   'Skew', 'Kurtosis'], columns=self.annual_statistics.columns)

        for i in range(len(self.annual_statistics.columns)):
            row = self.result_df.iloc[:, i].astype(float).dropna()
            period_stats.iloc[0, i] = np.mean(row)
            period_stats.iloc[1, i] = np.std(row)
            period_stats.iloc[2, i] = np.std(row, ddof=1) / np.sqrt(len(row))
            period_stats.iloc[4, i] = np.min(row)
            period_stats.iloc[5, i] = np.percentile(row, 25)
            period_stats.iloc[6, i] = np.percentile(row, 50)
            period_stats.iloc[7, i] = np.percentile(row, 75)
            period_stats.iloc[8, i] = np.max(row)
            period_stats.iloc[9, i] = row.skew()
            period_stats.iloc[10, i] = row.kurtosis()

            # Statistical significance test
            X = np.ones(len(row))
            y = row.values
            model = sm.OLS(y, X)
            newey_west = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            period_stats.iloc[3, i] = np.round(newey_west.tvalues[0], 4)

        self.monthly_statistics = period_stats

    def save_results(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            
        self.annual_statistics.to_csv(
            os.path.join(file_path, 'annual_statistics.csv'))
        self.monthly_statistics.to_csv(
            os.path.join(file_path, 'period_statistics.csv'))

    def get_frequency_info(self):
        """Return information about the detected frequency"""
        return {
            'frequency': self.frequency,
            'periods_per_year': self.periods_per_year,
            'data_length': len(self.result_df),
            'date_range': f"{self.result_df.index[0]} to {self.result_df.index[-1]}"
        }


if __name__ == '__main__':
    result_df = pd.read_csv('../general_results/result.csv', index_col=0)
    M = METRICS(result_df)
    M.cal_describe()
    M.cal_monthly_statistics()
    M.save_results('../general_results')

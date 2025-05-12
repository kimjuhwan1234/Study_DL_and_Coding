import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

pd.set_option('display.precision', 5)

class METRICS:
    def __init__(self, result_df: pd.DataFrame, period: pd.Index):
        self.result_df = result_df.loc[period, :].dropna(axis=0)
        self.annual_statistics = pd.DataFrame()

    def cal_describe(self):
        for col in (self.result_df.columns):
            statistic_dict = {}
            row = self.result_df.loc[:, col]

            statistic_dict['count'] = len(self.result_df.index)
            statistic_dict['cumulative return'] = np.exp(np.mean(row) * 12) - 1
            statistic_dict['annualized return mean'] = np.exp(np.mean(row) * 12) - 1
            statistic_dict['annualized return std'] = np.exp(np.std(row) * np.sqrt(12)) - 1
            statistic_dict['annualized return downside std'] = (np.exp(np.std(row[row < 0]) * np.sqrt(12)) - 1) if len(
                row[row < 0]) > 1 else 0.01
            statistic_dict['sharpe ratio'] = (np.exp(np.mean(row) * 12) - 1) / statistic_dict['annualized return std']
            statistic_dict['sortino ratio'] = (np.exp(np.mean(row) * 12) - 1) / statistic_dict[
                'annualized return downside std']
            statistic_dict['gross profit'] = row[row > 0].sum()
            statistic_dict['gross loss'] = row[row < 0].sum()
            statistic_dict['profit factor'] = statistic_dict['gross profit'] / np.abs(statistic_dict['gross loss'])
            statistic_dict['maximum drawdown'] = self.cal_MDD(row)
            statistic_dict['calmar ratio'] = self.cal_calmar_ratio(row)
            statistic_dict['profitable years'] = self.cal_profit_loss_years(row, True)
            statistic_dict['unprofitable years'] = self.cal_profit_loss_years(row, False)

            statistic_df = pd.DataFrame(data=statistic_dict.values(), index=statistic_dict.keys(), columns=[col])
            self.annual_statistics = pd.concat([self.annual_statistics, statistic_df], axis=1)

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

        calmar = (np.exp(np.mean(row) * 12) - 1) / abs(max_drawdown)
        return calmar

    def cal_profit_loss_years(self, row, profit):
        row.index = pd.to_datetime(row.index)
        row2 = pd.DataFrame(row.index.year, index=row.index)
        year_df = pd.concat([row, row2], axis=1)
        sum_by_year = year_df.groupby(year_df.index.year)[year_df.columns[0]].sum()
        if profit:
            return sum_by_year[sum_by_year > 0].count()
        else:
            return sum_by_year[sum_by_year < 0].count()

    def cal_t_statistics(self):
        t_test = pd.DataFrame(index=['t-statistic', 'p_value'], columns=self.annual_statistics.columns)
        for i in range(len(self.annual_statistics.columns)):
            t_statistic, p_value = stats.ttest_ind_from_stats(self.annual_statistics.iloc[2, i],
                                                              self.annual_statistics.iloc[3, i],
                                                              self.annual_statistics.iloc[0, i],
                                                              self.annual_statistics.iloc[2, -1],
                                                              self.annual_statistics.iloc[3, -1],
                                                              self.annual_statistics.iloc[0, -1])
            t_test.iloc[0, i] = t_statistic
            t_test.iloc[1, i] = p_value

        self.annual_statistics = pd.concat([self.annual_statistics, t_test], axis=0)

    def cal_monthly_statistics(self):
        month = pd.DataFrame(
            index=['Mean', 'Standard deviation', 'Standard error', 't-statistic', 'Min', '25%', '50%', '75%', 'Max',
                   'Skew', 'Kurtosis'], columns=self.annual_statistics.columns)

        for i in range(len(self.annual_statistics.columns)):
            month.iloc[0, i] = np.mean(self.result_df.iloc[:, i])
            month.iloc[1, i] = np.std(self.result_df.iloc[:, i])
            month.iloc[2, i] = np.std(self.result_df.iloc[:, i], ddof=1) / np.sqrt(len(self.result_df.iloc[:, i]))
            month.iloc[4, i] = np.min(self.result_df.iloc[:, i])
            month.iloc[5, i] = np.percentile(self.result_df.iloc[:, i], 25)
            month.iloc[6, i] = np.percentile(self.result_df.iloc[:, i], 50)
            month.iloc[7, i] = np.percentile(self.result_df.iloc[:, i], 75)
            month.iloc[8, i] = np.max(self.result_df.iloc[:, i])
            month.iloc[9, i] = self.result_df.iloc[:, i].skew()
            month.iloc[10, i] = self.result_df.iloc[:, i].kurtosis()

            # X = sm.add_constant(self.result_df.iloc[i, :].shift(1).dropna())
            X = np.ones(len(self.result_df))
            # y = self.result_df.iloc[i, :][1:].dropna().values
            y = self.result_df.iloc[:, i].values
            # y.index = X.index
            model = sm.OLS(y, X)
            newey_west = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            month.iloc[3, i] = np.round(newey_west.tvalues[0], 4)

        self.monthly_statistics = month

    def save_results(self, file_path):
        self.annual_statistics.applymap(lambda x: round(x, 4)).to_csv(os.path.join(file_path, 'annual_statistics.csv'))
        self.monthly_statistics.applymap(lambda x: round(x, 4)).to_csv(os.path.join(file_path, 'monthly_statistics.csv'))


if __name__ == '__main__':
    result_df = pd.read_csv('../general_results/result.csv', index_col=0)
    M = METRICS(result_df, result_df.index)
    M.cal_describe()
    M.cal_monthly_statistics()
    M.save_results('../general_results')

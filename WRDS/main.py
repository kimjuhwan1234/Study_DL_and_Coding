import pandas as pd
import os
import common as c


def checkdir():
    for folder in ['input', 'output_direct', 'output_dropna',
                   'output_dropna/simple', 'output_dropna/smart', 'output_dropna/smartexmom']:
        if not os.path.exists(folder):
            os.makedirs(folder)


def green2output():
    # Convert Green's file(var `temp2` in the sas code) into our format (files separated by date)
    chars = pd.read_csv(os.path.join('input', '_characteristics.csv'))[
        'name'].tolist()
    necessary_cols = [item.lower() for item in chars.copy() + ['gvkey', 'DATE', 'prc']]

    df = pd.read_csv(os.path.join('input', 'green.csv'), low_memory=False)
    df = df[df.columns[df.columns.str.lower().isin(necessary_cols)]]
    df = df.dropna(subset=['gvkey', 'DATE']).drop(index=163354)  # somehow that row is duplicated
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d').dt.strftime('%Y-%m-%d')

    # pivot the price dataframe
    prc_df = df[['gvkey', 'DATE', 'prc']]
    prc_df = prc_df.pivot(index='DATE', columns='gvkey', values='prc').sort_index()
    mom_df = prc_df / prc_df.shift(1) - 1
    """
    At this stage, mom_df looks like:
    >>> prc_df.iloc[-5:]
        gvkey       1001    1003       1004    ...  293754  311524     315318
        DATE                                   ...                           
        2023-08-31     NaN     NaN  61.599998  ...     NaN     NaN  20.620001
        2023-09-29     NaN     NaN  59.529999  ...     NaN     NaN  19.610001
        2023-10-31     NaN     NaN  59.360001  ...     NaN     NaN  18.230000
        2023-11-30     NaN     NaN  69.300003  ...     NaN     NaN  20.959999
        2023-12-29     NaN     NaN  62.400002  ...     NaN     NaN  23.139999
        [5 rows x 18430 columns]
    >>> mom_df.iloc[-5:]
        gvkey       1001    1003      1004    ...  293754  311524    315318
        DATE                                  ...                          
        2023-08-31     NaN     NaN  0.030100  ...     NaN     NaN -0.016221
        2023-09-29     NaN     NaN -0.033604  ...     NaN     NaN -0.048982
        2023-10-31     NaN     NaN -0.002856  ...     NaN     NaN -0.070372
        2023-11-30     NaN     NaN  0.167453  ...     NaN     NaN  0.149753
        2023-12-29     NaN     NaN -0.099567  ...     NaN     NaN  0.104008
        [5 rows x 18430 columns]
    Note that 62.400002/69.300003 - 1 = -0.099567, 59.360001/62.400002 - 1 = -0.002856, etc.
    That is, `YYYY-MM-DD` in mom_df contains the return from `YYYY-(MM-1)-DD` to `YYYY-MM-DD`.
    """
    mom_df.to_csv(os.path.join('output_direct', '_momentum.csv'), index=True)
    del mom_df

    # datadate column contains last days of the month
    all_dates = pd.Series(df['DATE'].unique()).sort_values()
    print(f'Converting Green\'s data\n{df['DATE'].unique()[0][:4]} [', end='')
    for date in all_dates.iloc[1:]:
        charm_df = df[df['DATE'] == date]
        charm_df = charm_df.drop(columns='DATE').set_index('gvkey')
        charm_df = charm_df.sort_index(axis=1).sort_index()

        # add mom/price data
        todays_index = prc_df.index.get_loc(date)
        last_month = prc_df.index[todays_index - 1]  # YYYY-(MM-1)-DD
        t48m_prc_df = prc_df.iloc[max(todays_index - 48, 0):todays_index]  # YYYY-(MM-48)-DD to YYYY-(MM-1)-DD
        momm_df = t48m_prc_df.iloc[-1]/t48m_prc_df - 1  # mom_n where n > 1
        momm_df.loc[last_month] = prc_df.iloc[todays_index]/prc_df.loc[last_month] - 1  # mom_1

        # rename columns
        ind_dict = pd.Series(momm_df.index).sort_values(ascending=False).reset_index(drop=True).to_dict()
        ind_dict = {ind_dict[i]: 'mom_' + str(i+1) for i in ind_dict}
        momm_df = momm_df.rename(index=ind_dict)
        momm_df = momm_df.dropna(axis=1, how='all').T
        """ At this stage, 1980-04-30 for example, the resulting momm_df looks like:
        >>> prc_df.loc['1980-01-31':date, :]
            gvkey       1001    1003    1004    1005    ...  289735  293754  311524  315318
            DATE                                        ...                                
            1980-01-31     NaN     NaN  14.375   18.50  ...     NaN     NaN     NaN     NaN
            1980-02-29     NaN     NaN  14.750   23.50  ...     NaN     NaN     NaN     NaN
            1980-03-31     NaN     NaN  10.375   18.25  ...     NaN     NaN     NaN     NaN
            1980-04-30     NaN     NaN  10.375   10.25  ...     NaN     NaN     NaN     NaN
            [4 rows x 18430 columns]
        >>> momm_df.T
            gvkey     1004      1005      1010   ...     27866     31596     65351
            DATE                                 ...                              
            mom_3 -0.278261 -0.013514 -0.231928  ... -0.292419       NaN -0.171779
            mom_2 -0.296610 -0.223404 -0.172078  ... -0.206478 -0.039216 -0.062500
            mom_1  0.000000 -0.438356 -0.027451  ...  0.122449  0.156463 -0.029630
            [3 rows x 3508 columns]
        Note that 10.375/10.375 - 1 = 0, 10.375/14.750 - 1 = -0.296610, 10.375/14.375 - 1 = -0.278261, etc.
        That is, mom_1 contains the previous 1 month return,
        mom_n contains the return from n months ago to 1 month ago.
        """

        charm_df = charm_df.join(momm_df, how='left')
        charm_df.to_csv(os.path.join('output_direct', date + '.csv'), index=True)

        print('-', end='')
        print(f']\n{int(date[:4])+1} [', end='') if '-12-' in date else None

    print('\n\nOverall NaN ratio:', round(df.isna().sum().sum() / df.size * 100, 2), '%\n\n')
    del df, prc_df


def direct2dropnas():
    start_year = c.listdir('output_direct')[0][:4]
    print(f'Dropping NaNs\n{start_year} [', end='')
    for afile in c.listdir('output_direct', '.csv'):
        if afile.startswith('_'):
            continue
        df = pd.read_csv(os.path.join('output_direct', afile)).set_index('gvkey')
        df_simple = df.dropna(axis=1, how='any')
        df_simple.to_csv(os.path.join('output_dropna', 'simple', afile), index=True)
        df_smart = c.smartdropna(df)
        df_smart.to_csv(os.path.join('output_dropna', 'smart', afile), index=True)
        df_smart2 = c.smartdropna(df, exclude=['mom_'+str(i) for i in range(1, 49)])
        df_smart2.to_csv(os.path.join('output_dropna', 'smartexmom', afile), index=True)

        print('-', end='')
        print(f']\n{int(afile[:4]) + 1} [', end='') if '-12-' in afile else None


if __name__ == '__main__':
    checkdir()
    # green2output()
    direct2dropnas()

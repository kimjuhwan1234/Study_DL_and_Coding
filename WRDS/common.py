import os
import pandas as pd


def listdir(path, ext='.csv'):
    ret = [f for f in os.listdir(path) if f.endswith(ext)]
    ret.sort()
    return ret


def smartdropna(df: pd.DataFrame, exclude=None) -> pd.DataFrame:
    exclude = [item for item in exclude if item in df.columns] if exclude else []
    mask = df.notna()  # smaller(memory-allocation-wise) dataframe
    mask = mask.drop(columns=exclude, errors='ignore')
    best_mask = pd.DataFrame()
    for thresh_col in range(10, 1, -1):  # 1 = 10% filled
        # drop columns with too many False
        mask_col = mask[mask.columns[mask.sum(axis=0) > (mask.shape[0] * thresh_col / 10)]]
        if mask_col.empty:
            continue
        for thresh_row in range(10, 1, -1):
            # drop rows with too many False
            mask_colrow = mask_col.loc[mask_col.index[mask_col.sum(axis=1) > (mask_col.shape[1] * thresh_row / 10)]]
            if mask_colrow.empty:
                continue
            # drop columns with any False
            mask_colrow = mask_colrow[mask_colrow.columns[mask_colrow.all(axis=0)]]
            if mask_colrow.empty:
                continue

            # if the current mask is better than the previous one, update the best mask
            if mask_colrow.size > best_mask.size:
                best_mask = mask_colrow
                # probably we can implement a break here
                # but due to the lack of my compsci knowledge,
                # I will rather take advantage of my computer's speed
    ret = pd.concat([df[best_mask.columns], df[exclude]], axis=1, sort=False).dropna()
    return ret

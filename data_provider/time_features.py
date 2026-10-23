"""
作用：生成与官方 forecasting 工具一致的连续日历时间特征。
冻结状态：正式 baseline 开始后硬冻结。
修改影响：任何修改都必须重跑全部 baseline 和后续模型。
"""

import numpy as np
import pandas as pd


def time_features(dates, frequency):
    """Return [length, time_dim] float32 features in approximately [-0.5, 0.5]."""
    index = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise"))
    if frequency == "monthly":
        values = [(index.month - 1) / 11.0 - 0.5]
    elif frequency == "weekly":
        iso_week = index.isocalendar().week.to_numpy(dtype=np.float64)
        values = [(index.day - 1) / 30.0 - 0.5, (iso_week - 1) / 52.0 - 0.5]
    elif frequency == "daily":
        values = [
            index.dayofweek / 6.0 - 0.5,
            (index.day - 1) / 30.0 - 0.5,
            (index.dayofyear - 1) / 365.0 - 0.5,
        ]
    else:
        raise ValueError("Unsupported frequency {!r}".format(frequency))
    return np.vstack(values).T.astype(np.float32, copy=False)

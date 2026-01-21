# 工程化模板
from __future__ import annotations

# 载入必要的包
import numpy as np
import pandas as pd

########################################
############## rsi指标 #################
########################################

def add_rsi(df:pd.DataFrame, close_col:str, window:int = 14, col_name: str | None = None):
    '''
    Docstring for add_rsi
    不用连续复利，会更能体现大单的市场力量
    :param df: 原始数据
    :type df: pd.DataFrame
    :param close_col: 确定close的那一列是什么
    :type close_col: str
    :param window: RSI的计算周期
    :type window: int
    :param col_name: 可以指定这个指标的名称
    '''
    d = df.copy()
    delta = d[close_col].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    name = col_name or f"RSI{window}"
    d[name] = 100 - (100 / (1 + rs))
    return d
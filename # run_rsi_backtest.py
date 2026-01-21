# run_rsi_backtest.py
import pandas as pd

from MiniQuant import add_rsi, RSIIntradayBacktest
from MiniQuant.report import plot_rsi_backtest

def main():
    df = pd.read_csv("rb_5min.csv")  # 你自己的数据文件
    prefix = "KQ.m@SHFE.rb"                    # 取决于你原始列名的 prefix

    # 1) prepare + indicator
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d = add_rsi(d, close_col=f"{prefix}.close", window=14, col_name="RSI14")

    # 2) run backtest
    bt = RSIIntradayBacktest(d, symbol_prefix=prefix, rsi_col="RSI14")
    bt.run()
    summary = bt.metrics()
    print("SUMMARY:", summary)

    # 3) plot
    title = f"{prefix} | Final={summary['final_equity']:.2f} | MDD={summary['max_drawdown']:.2%} | Sharpe={summary['sharpe']:.3f}"
    fig = plot_rsi_backtest(bt.df, bt.trades_df, bt.rsi_col, bt.rsi_long, bt.rsi_short, title=title)
    fig.show()

if __name__ == "__main__":
    main()
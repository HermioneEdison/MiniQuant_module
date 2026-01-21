# miniquant/report.py
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_rsi_backtest(df: pd.DataFrame, trades_df: pd.DataFrame | None, rsi_col: str, rsi_long: float, rsi_short: float, title: str = "",
                      reveal_close = False):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        specs=[[{"type": "candlestick"}], [{"type": "scatter"}], [{"type": "scatter"}]],
    )

    fig.add_trace(go.Candlestick(
        x=df["datetime"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Kline"
    ), row=1, col=1)

    if trades_df is not None and len(trades_df) > 0:
        t = trades_df.copy()
        t["datetime"] = pd.to_datetime(t["datetime"])
        if reveal_close:
            for action in ["BUY", "SELL", "CLOSE"]:
                sub = t[t["action"] == action]
                if len(sub) > 0:
                    fig.add_trace(go.Scatter(x=sub["datetime"], y=sub["price"], mode="markers", name=action), row=1, col=1)
        else:
            for action in ["BUY", "SELL"]:
                sub = t[t["action"] == action]
                if len(sub) > 0:
                    fig.add_trace(go.Scatter(x=sub["datetime"], y=sub["price"], mode="markers", name=action), row=1, col=1)

    if rsi_col in df.columns:
        fig.add_trace(go.Scatter(x=df["datetime"], y=df[rsi_col], mode="lines", name=rsi_col), row=2, col=1)
        fig.add_hline(y=rsi_long, line_dash="dash", row=2, col=1)
        fig.add_hline(y=rsi_short, line_dash="dash", row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    fig.add_trace(go.Scatter(x=df["datetime"], y=df["cum_pnl"], mode="lines", name="CumPnL"), row=3, col=1)

    fig.update_layout(
        height=900,
        title=title,
        hovermode="x unified",
        margin=dict(l=30, r=20, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
    fig.update_xaxes(rangeslider_visible=True, row=3, col=1)

    return fig
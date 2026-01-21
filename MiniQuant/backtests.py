# miniquant/backtests.py
from __future__ import annotations
import numpy as np
import pandas as pd

class RSIIntradayBacktest:
    def __init__(
        self,
        df: pd.DataFrame,
        symbol_prefix: str,
        rsi_col: str = "RSI14",
        initial_cap: float = 1_000_000.0,
        stop_loss_pct: float = 0.01,
        rsi_long: float = 70.0,
        rsi_short: float = 40.0,
        max_entries_per_day: int = 5,
        max_exits_per_day: int = 5,
    ):
        self.df_raw = df
        self.symbol_prefix = symbol_prefix
        self.rsi_col = rsi_col

        self.initial_cap = initial_cap
        self.stop_loss_pct = stop_loss_pct
        self.rsi_long = rsi_long
        self.rsi_short = rsi_short
        self.max_entries_per_day = max_entries_per_day
        self.max_exits_per_day = max_exits_per_day

        self.df: pd.DataFrame | None = None
        self.trades: list[dict] = []
        self.trades_df: pd.DataFrame | None = None
        self.summary: dict | None = None

    def prepare_df(self) -> "RSIIntradayBacktest":
        d = self.df_raw.copy()
        d["datetime"] = pd.to_datetime(d["datetime"])
        d = d.sort_values("datetime").reset_index(drop=True)

        prefix = self.symbol_prefix
        rename_map = {
            f"{prefix}.open": "open",
            f"{prefix}.high": "high",
            f"{prefix}.low": "low",
            f"{prefix}.close": "close",
            f"{prefix}.volume": "volume",
            f"{prefix}.open_oi": "open_oi",
            f"{prefix}.close_oi": "close_oi",
        }
        d = d.rename(columns={k: v for k, v in rename_map.items() if k in d.columns})

        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in d.columns]
        if missing:
            raise ValueError(f"OHLC rename failed, missing: {missing}. Check symbol_prefix={prefix}")

        d["date"] = d["datetime"].dt.date
        self.df = d
        return self

    def run(self) -> "RSIIntradayBacktest":
        if self.df is None:
            self.prepare_df()

        df = self.df.copy()

        pos = 0
        entry_price = np.nan
        cash = self.initial_cap

        entries_today = 0
        exits_today = 0
        current_date = None

        equity = []
        pre_rsi = np.nan

        def record(ts, action, price, qty, pos_after, reason, cash_=None, rsi_value=None, pnl_log=None, pnl_cash=None):
            self.trades.append({
                "datetime": ts,
                "action": action,
                "price": float(price),
                "qty": int(qty),
                "pos_after": int(pos_after),
                "reason": reason,
                "rsi": None if rsi_value is None else float(rsi_value),
                "PnL_log": None if pnl_log is None else float(pnl_log),
                "PnL_cash": None if pnl_cash is None else float(pnl_cash),
                "cash_after": None if cash_ is None else float(cash_),
            })

        def mark_to_market(cur_close: float) -> float:
            if pos == 0 or not np.isfinite(entry_price):
                return cash
            if pos == 1:
                ret = np.log(cur_close) - np.log(entry_price)
            else:
                ret = np.log(entry_price) - np.log(cur_close)
            return cash * np.exp(ret)

        for i, row in df.iterrows():
            ts = row["datetime"]
            date = row["date"]
            open_ = float(row["open"])
            close = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            cur_rsi = row.get(self.rsi_col, np.nan)

            can_trade_signal = np.isfinite(cur_rsi)
            stopped_out = False

            if current_date is None:
                current_date = date
            elif date != current_date:
                entries_today = 0
                exits_today = 0
                current_date = date

            # stop loss
            if pos != 0 and np.isfinite(entry_price) and (exits_today < self.max_exits_per_day):
                if pos == 1 and low <= entry_price * (1 - self.stop_loss_pct):
                    cash_before = cash
                    loss = np.log(close) - np.log(entry_price)
                    cash *= np.exp(loss)
                    pnl_cash = cash - cash_before
                    exits_today += 1
                    pos = 0
                    entry_price = np.nan
                    stopped_out = True
                    record(ts, "SELL", close, 1, pos, "stop_long_loss", cash_=cash, rsi_value=pre_rsi, pnl_cash=pnl_cash)

                elif pos == -1 and high >= entry_price * (1 + self.stop_loss_pct):
                    cash_before = cash
                    loss = np.log(entry_price) - np.log(close)
                    cash *= np.exp(loss)
                    pnl_cash = cash - cash_before
                    exits_today += 1
                    pos = 0
                    entry_price = np.nan
                    stopped_out = True
                    record(ts, "BUY", close, 1, pos, "stop_short_loss", cash_=cash, rsi_value=pre_rsi, pnl_cash=pnl_cash)

            # entries
            if pos == 0 and entries_today < self.max_entries_per_day and (not stopped_out) and np.isfinite(pre_rsi) and can_trade_signal:
                if pre_rsi >= self.rsi_long:
                    pos = 1
                    entry_price = open_
                    entries_today += 1
                    record(ts, "BUY", open_, 1, pos, "LONG", rsi_value=pre_rsi)

                elif pre_rsi <= self.rsi_short:
                    pos = -1
                    entry_price = open_
                    entries_today += 1
                    record(ts, "SELL", open_, 1, pos, "SHORT", rsi_value=pre_rsi)

            pre_rsi = cur_rsi

            is_last_bar = (i == len(df) - 1) or (df.loc[i + 1, "date"] != date)
            if is_last_bar and pos != 0:
                cash_before = cash
                if pos == 1:
                    ret = np.log(close) - np.log(entry_price)
                else:
                    ret = np.log(entry_price) - np.log(close)
                pos = 0
                cash *= np.exp(ret)
                pnl_cash = cash - cash_before
                entry_price = np.nan
                record(ts, "CLOSE", close, 1, pos, "END_TODAY", cash_=cash, rsi_value=pre_rsi, pnl_log=ret, pnl_cash=pnl_cash)

            equity.append(mark_to_market(close))

        df["equity"] = equity
        df["cum_pnl"] = df["equity"] - self.initial_cap
        df["ret"] = df["equity"].pct_change().fillna(0.0)

        self.df = df
        self.trades_df = pd.DataFrame(self.trades)
        return self

    def metrics(self) -> dict:
        if self.df is None or "equity" not in self.df.columns:
            raise RuntimeError("Run .run() first.")

        d = self.df
        peak = d["equity"].cummax()
        dd = d["equity"] / peak - 1.0
        max_dd = float(dd.min())

        bars_per_year = 48 * 250  # 5min bars approx
        mu = d["ret"].mean()
        sig = d["ret"].std(ddof=0)
        sharpe = float((mu / sig) * np.sqrt(bars_per_year)) if sig > 0 else np.nan

        self.summary = {
            "initial_cap": self.initial_cap,
            "final_equity": float(d["equity"].iloc[-1]),
            "cum_pnl": float(d["cum_pnl"].iloc[-1]),
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "num_trades": 0 if self.trades_df is None else int(len(self.trades_df)),
        }
        return self.summary
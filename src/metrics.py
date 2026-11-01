import numpy as np
import pandas as pd

TRADING_DAYS = 252


# Compound daily returns, then scale the growth to a full trading year.
def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if len(returns) == 0:
        return np.nan
    total_growth = (1 + returns).prod()
    return total_growth ** (periods_per_year / len(returns)) - 1

# Convert the standard deviation of daily returns into annual volatility.
def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if len(returns) < 2:
        return np.nan
    return returns.std() * np.sqrt(periods_per_year)


# annualized Sharpe ratio, assuming a zero risk-free rate. Short sales mostly fund the longs, so
# little cash is tied up. Unequal leg sizes and borrow fees make this slightly optimistic.
def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    daily_std = returns.std() if len(returns) >= 2 else np.nan
    # a flat return series has zero std, so the Sharpe ratio is undefined.
    if not daily_std > 0:
        return np.nan
    return returns.mean() / daily_std * np.sqrt(periods_per_year)


# largest fall from a running peak of the equity curve, as a negative fraction (-0.2 is a 20%
# drawdown). The curve starts at 1, so a loss on the first day already counts as a drawdown.
def max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return np.nan
    equity = (1 + returns).cumprod()
    running_peak = equity.cummax().clip(lower=1.0)
    return (equity / running_peak - 1).min()


# split each pair's daily results into round-trip trades. A trade opens on the day the position
# turns nonzero and closes on the first flat day after it: that day still earns the return of the
# position held overnight and pays the exit cost, so it belongs to the trade. Positions never flip
# directly from long to short, so trades never share a day.
def extract_trades(pair_results: pd.DataFrame) -> pd.DataFrame:
    trades = []
    for (window_id, pair), rows in pair_results.groupby(["window", "pair"], sort=False):
        rows = rows.sort_values("date")
        positions = rows["position"].to_numpy()
        is_open = positions != 0
        # Shift the open/closed status by one day to see if yesterday had a position.
        was_open = np.r_[False, is_open[:-1]]
        # each entry starts a new trade id, and the exit day keeps the id of the trade it closes.
        trade_ids = np.cumsum(is_open & ~was_open)
        in_trade = is_open | was_open
        for trade_id in np.unique(trade_ids[in_trade]):
            trade_rows = rows[in_trade & (trade_ids == trade_id)]
            trades.append({
                "window": window_id,
                "period": trade_rows["period"].iloc[0] if "period" in trade_rows else None,
                "pair": pair,
                "entry_date": trade_rows["date"].iloc[0],
                "exit_date": trade_rows["date"].iloc[-1],
                "direction": int(trade_rows["position"].iloc[0]),
                "holding_days": int((trade_rows["position"] != 0).sum()),
                "trade_return": (1 + trade_rows["net_return"]).prod() - 1,
            })
    return pd.DataFrame(
        trades,
        columns=["window", "period", "pair", "entry_date", "exit_date", "direction",
                 "holding_days", "trade_return"],
    )


# headline numbers for one daily portfolio return series and the trades behind it.
def performance_summary(returns: pd.Series, trades: pd.DataFrame | None = None) -> pd.Series:
    summary = {
        "annual_return": annualized_return(returns),
        "annual_volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "total_return": (1 + returns).prod() - 1,
    }
    if trades is not None:
        has_trades = len(trades) > 0
        summary["n_trades"] = len(trades)
        # share of trades that made money after costs.
        summary["hit_rate"] = (trades["trade_return"] > 0).mean() if has_trades else np.nan
        summary["avg_trade_return"] = trades["trade_return"].mean() if has_trades else np.nan
        summary["avg_holding_days"] = trades["holding_days"].mean() if has_trades else np.nan
    return pd.Series(summary)

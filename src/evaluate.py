import argparse

import pandas as pd

from src.backtest import walk_forward, walk_forward_windows
from src.data_loader import PROJECT_ROOT, load_config, load_data, load_universe, to_log_prices
from src.metrics import extract_trades, performance_summary
from src.portfolio import equal_weight_returns

TABLES_DIR = PROJECT_ROOT / "results" / "tables"


# every trading day in one period's windows, including days when no pair was traded.
def period_trading_dates(
    period: str, windows: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    period_windows = windows[windows["period"] == period]
    start, end = period_windows["trading_start"].iloc[0], period_windows["trading_end"].iloc[-1]
    return dates[(dates >= start) & (dates <= end)]


# performance of one period (validation or holdout): the equal-weight portfolio over every trading
# day in that period's windows, net and gross of costs, plus the trades behind it.
def period_summary(
    period: str, windows: pd.DataFrame, pair_results: pd.DataFrame, log_prices: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    trading_dates = period_trading_dates(period, windows, log_prices.index)
    period_results = pair_results[pair_results["period"] == period]
    net_returns = equal_weight_returns(period_results, trading_dates, "net_return")
    gross_returns = equal_weight_returns(period_results, trading_dates, "gross_return")
    trades = extract_trades(period_results)
    summary = pd.DataFrame({
        "net": performance_summary(net_returns, trades),
        "gross": performance_summary(gross_returns),
    })
    return summary, net_returns, trades


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # the holdout is reported once, after every setting is frozen on validation.
    parser.add_argument("--holdout", action="store_true", help="also report the holdout period")
    args = parser.parse_args()

    cfg = load_config()
    wf_cfg = cfg["walk_forward"]
    prices, _, _ = load_data(cfg)
    log_prices = to_log_prices(prices)
    windows = walk_forward_windows(
        log_prices.index, wf_cfg["formation_months"], wf_cfg["trading_months"],
        wf_cfg["validation_end"],
    )
    pair_results, _ = walk_forward(
        log_prices, load_universe(cfg), **wf_cfg,
        pair_params=cfg["pairs"], signal_params=cfg["signals"], **cfg["backtest"],
    )

    periods = ["validation", "holdout"] if args.holdout else ["validation"]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for period in periods:
        summary, net_returns, trades = period_summary(period, windows, pair_results, log_prices)
        print(f"\n{period}\n{summary.to_string(float_format='{:.4f}'.format)}")
        summary.to_csv(TABLES_DIR / f"{period}_summary.csv")
        net_returns.to_csv(TABLES_DIR / f"{period}_daily_returns.csv")
        trades.to_csv(TABLES_DIR / f"{period}_trades.csv", index=False)

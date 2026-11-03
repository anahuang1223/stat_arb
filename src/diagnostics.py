import numpy as np
import pandas as pd

from src.backtest import walk_forward, walk_forward_windows
from src.data_loader import PROJECT_ROOT, load_config, load_data, load_universe, to_log_prices
from src.evaluate import TABLES_DIR, period_trading_dates
from src.metrics import annualized_return, extract_trades, sharpe_ratio
from src.portfolio import equal_weight_returns
from src.statistical_tests import engle_granger, half_life
from src.visualization import plot_equity_curves, plot_pvalue_histogram, plot_spread_examples

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


# how many pairs survive each stage of selection in every window. If no pair were cointegrated,
# about 5% of the tests would still have a raw p-value below 0.05, so n_raw_p05 near
# n_expected_by_chance means the tests found little real evidence. The last two columns show
# whether the traded pairs are separate bets or several pairs on the same stock.
def screening_summary(scored_pairs: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window_id, window in windows.iterrows():
        pairs = scored_pairs[scored_pairs["window"] == window_id]
        traded = pairs[pairs["traded"]]
        stocks = pd.concat([traded["y"], traded["x"]])
        rows.append({
            "window": window_id,
            "period": window.period,
            "trading_start": window.trading_start,
            "n_tested": len(pairs),
            "n_raw_p05": int((pairs["pvalue"] < 0.05).sum()),
            "n_expected_by_chance": 0.05 * len(pairs),
            "n_bh_significant": int(pairs["significant"].sum()),
            "n_selected": int(pairs["selected"].sum()),
            "n_traded": len(traded),
            "n_distinct_stocks": stocks.nunique(),
            # 3 means one stock appears in three of the traded pairs.
            "max_pairs_per_stock": int(stocks.value_counts().max()) if len(stocks) else 0,
        })
    return pd.DataFrame(rows)


# label why each trade closed, from the z-score on its exit day: "stop" if the spread moved past
# stop_z against the position, "reverted" if it came back within exit_z, and "window_end" if
# neither happened and the trade was closed on the last day of its trading window.
def label_exits(
    trades: pd.DataFrame, pair_results: pd.DataFrame, exit_z: float, stop_z: float
) -> pd.DataFrame:
    zscore_by_day = pair_results.set_index(["window", "pair", "date"])["zscore"]
    # Match each trade to its z-score on the day it closed.
    exit_keys = list(zip(trades["window"], trades["pair"], trades["exit_date"]))
    exit_zscores = zscore_by_day.loc[exit_keys].to_numpy()
    # Multiply by trade direction so the same exit rules work for long and short spreads.
    signed_zscores = trades["direction"].to_numpy() * exit_zscores
    exit_reason = np.select(
        [signed_zscores <= -stop_z, signed_zscores >= -exit_z], ["stop", "reverted"],
        default="window_end",
    )
    return trades.assign(exit_zscore=exit_zscores, exit_reason=exit_reason)


# check whether each traded pair still behaves like a cointegrated pair while it is traded: the
# Engle-Granger p-value re-fit on the trading period, and the half-life, mean, and largest move of
# the trading spread built with the formation hedge ratio. Six months is little data for the test,
# so a high trading p-value is weak evidence on its own; a mean z-score far from 0 means the
# spread shifted to a new level.
def trading_stability(
    scored_pairs: pd.DataFrame, windows: pd.DataFrame, log_prices: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for pair, row in scored_pairs[scored_pairs["traded"]].iterrows():
        window = windows.loc[row["window"]]
        trading = log_prices.loc[window.trading_start:window.trading_end]
        spread = trading[row["y"]] - row["alpha"] - row["beta"] * trading[row["x"]]
        zscores = spread / row["spread_std"]
        rows.append({
            "window": row["window"],
            "period": row["period"],
            "pair": pair,
            "formation_pvalue": row["pvalue"],
            "trading_pvalue": engle_granger(trading[row["y"]], trading[row["x"]]),
            "formation_half_life": row["half_life"],
            "trading_half_life": half_life(spread),
            "trading_mean_zscore": zscores.mean(),
            "trading_max_abs_zscore": zscores.abs().max(),
        })
    return pd.DataFrame(rows)


# Sharpe ratio and annual return of each period's portfolio at several cost levels. Turnover does
# not depend on the cost, so net returns are rebuilt from gross returns without a new backtest.
def cost_sensitivity(
    pair_results: pd.DataFrame, windows: pd.DataFrame, dates: pd.DatetimeIndex,
    cost_grid: list[float],
) -> pd.DataFrame:
    rows = []
    for period in windows["period"].unique():
        trading_dates = period_trading_dates(period, windows, dates)
        period_results = pair_results[pair_results["period"] == period]
        for cost_bps in cost_grid:
            repriced = period_results.assign(
                net_return=period_results["gross_return"]
                - period_results["turnover"] * cost_bps / 10000
            )
            returns = equal_weight_returns(repriced, trading_dates)
            rows.append({
                "period": period,
                "cost_bps": cost_bps,
                "sharpe": sharpe_ratio(returns),
                "annual_return": annualized_return(returns),
            })
    return pd.DataFrame(rows)


# z-scores of one traded pair over its formation and trading periods, for the spread plots.
def spread_example(
    pair: str, window_id: int, scored_pairs: pd.DataFrame, windows: pd.DataFrame,
    log_prices: pd.DataFrame, label: str,
) -> dict:
    row = scored_pairs[(scored_pairs["window"] == window_id) & scored_pairs["traded"]].loc[pair]
    window = windows.loc[window_id]
    history = log_prices.loc[window.formation_start:window.trading_end]
    spread = history[row["y"]] - row["alpha"] - row["beta"] * history[row["x"]]
    return {
        "zscores": spread / row["spread_std"],
        "trading_start": window.trading_start,
        "title": f"{pair}, {window.period} window {window_id} ({label})",
    }


if __name__ == "__main__":
    cfg = load_config()
    wf_cfg, signal_cfg = cfg["walk_forward"], cfg["signals"]
    prices, _, _ = load_data(cfg)
    log_prices = to_log_prices(prices)
    windows = walk_forward_windows(
        log_prices.index, wf_cfg["formation_months"], wf_cfg["trading_months"],
        wf_cfg["validation_end"],
    )
    pair_results, scored_pairs = walk_forward(
        log_prices, load_universe(cfg), **wf_cfg,
        pair_params=cfg["pairs"], signal_params=signal_cfg, **cfg["backtest"],
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    screening = screening_summary(scored_pairs, windows)
    screening.to_csv(TABLES_DIR / "window_screening.csv", index=False)
    print(screening.to_string(index=False, float_format="{:.1f}".format))

    trades = label_exits(
        extract_trades(pair_results), pair_results, signal_cfg["exit_z"], signal_cfg["stop_z"]
    )
    trades.to_csv(TABLES_DIR / "trade_exits.csv", index=False)
    print("\nexit reasons\n" + pd.crosstab(trades["period"], trades["exit_reason"]).to_string())
    print("\naverage trade return by exit reason\n" + trades.pivot_table(
        index="period", columns="exit_reason", values="trade_return"
    ).round(4).to_string())

    stability = trading_stability(scored_pairs, windows, log_prices)
    stability.to_csv(TABLES_DIR / "pair_stability.csv", index=False)
    print("\nmedian stability of traded pairs\n" + stability.groupby("period")[[
        "formation_pvalue", "trading_pvalue", "formation_half_life", "trading_half_life",
        "trading_mean_zscore", "trading_max_abs_zscore",
    ]].median().round(3).to_string())

    costs = cost_sensitivity(pair_results, windows, log_prices.index, [0, 5, 10])
    costs.to_csv(TABLES_DIR / "cost_sensitivity.csv", index=False)
    print("\n" + costs.round(4).to_string(index=False))

    plot_pvalue_histogram(scored_pairs["pvalue"], FIGURES_DIR / "pvalue_histogram.png")

    curves = {}
    for period in windows["period"].unique():
        trading_dates = period_trading_dates(period, windows, log_prices.index)
        period_results = pair_results[pair_results["period"] == period]
        returns_dict = {}
        for column in ["net", "gross"]:
            returns_dict[column] = equal_weight_returns(
                period_results,
                trading_dates,
                f"{column}_return"
            )
        curves[period] = pd.DataFrame(returns_dict)
    plot_equity_curves(curves, FIGURES_DIR / "equity_curves.png")

    # one pair whose trades made the most, one that stopped out most often, and the holdout pair
    # that lost the most, so the plots show reversion, a broken relationship, and the holdout.
    pair_totals = trades.groupby(["period", "window", "pair"]).agg(
        total_return=("trade_return", lambda returns: (1 + returns).prod() - 1),
        n_stops=("exit_reason", lambda reasons: (reasons == "stop").sum()),
    ).reset_index()
    validation_pairs = pair_totals[pair_totals["period"] == "validation"]
    holdout_pairs = pair_totals[pair_totals["period"] == "holdout"]
    picks = [
        (validation_pairs.loc[validation_pairs["total_return"].idxmax()], "best validation pair"),
        (validation_pairs.loc[validation_pairs["n_stops"].idxmax()], "most stop-outs"),
        (holdout_pairs.loc[holdout_pairs["total_return"].idxmin()], "worst holdout pair"),
    ]
    examples = [
        spread_example(pick["pair"], pick["window"], scored_pairs, windows, log_prices,
                       f"{label}, {pick['total_return']:+.1%}")
        for pick, label in picks
    ]
    plot_spread_examples(examples, signal_cfg["entry_z"], signal_cfg["exit_z"],
                         signal_cfg["stop_z"], FIGURES_DIR / "spread_examples.png")

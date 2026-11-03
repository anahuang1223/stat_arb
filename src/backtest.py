import numpy as np
import pandas as pd

from src.pair_selection import select_pairs
from src.signals import formation_zscore, generate_positions


# turn spread positions into dollar weights on each leg. Long the spread is long Y and short
# beta * X, scaled so the gross exposure |w_y| + |w_x| is 1 whenever a position is open.
def pair_weights(positions: pd.Series, beta: float) -> pd.DataFrame:
    gross = 1 + abs(beta)
    return pd.DataFrame(
        {"weight_y": positions / gross, "weight_x": -positions * beta / gross},
        index=positions.index,
    )


# daily portfolio returns from leg weights decided at each close. A trade is filled at that same
# close (in live trading: signal a few minutes before the close, filled in the closing auction),
# so weights are shifted by one day and a signal on day t only earns returns from day t + 1.
# Transaction costs are charged on the day the trade happens, per unit of weight traded.
def portfolio_returns(
    prices: pd.DataFrame, weights: pd.DataFrame, cost_bps: float = 0.0
) -> pd.DataFrame:
    asset_returns = prices.pct_change().fillna(0.0).to_numpy()
    held_weights = weights.shift(1).fillna(0.0).to_numpy()
    gross_return = (held_weights * asset_returns).sum(axis=1)

    # turnover on the first day is the full opening weight, since the book starts flat.
    # turnover measures total absolute changes in asset weights (how much we trade), used to
    # calculate transaction costs.
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = turnover.to_numpy() * cost_bps / 10000

    return pd.DataFrame(
        {
            "gross_return": gross_return,
            "turnover": turnover.to_numpy(),
            "cost": cost,
            "net_return": gross_return - cost,
        },
        index=weights.index,
    )


# backtest one pair between start and end using a hedge ratio and spread std fit on earlier
# (formation) data, so nothing from the trading period leaks in. Positions are generated only
# from start onward, so no earlier trade carries in, and the book is flattened on the last day so
# no trade is left open past end.
def backtest_pair(
    log_prices: pd.DataFrame,
    y_ticker: str,
    x_ticker: str,
    alpha: float,
    beta: float,
    spread_std: float,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    after_stop: str = "reenter",
    cost_bps: float = 0.0,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    spread = log_prices[y_ticker] - alpha - beta * log_prices[x_ticker]
    spread = spread.loc[start:end]
    zscores = formation_zscore(spread, spread_std)
    positions = generate_positions(zscores, entry_z, exit_z, stop_z, after_stop)
    # Force the strategy to close any open position on the final trading day.
    positions.iloc[-1] = 0
    weights = pair_weights(positions, beta)
    # Convert log prices back to ordinary prices for percentage-return calculations.
    prices = np.exp(log_prices.loc[positions.index, [y_ticker, x_ticker]])
    returns = portfolio_returns(prices, weights, cost_bps)
    return pd.concat(
        [spread.rename("spread"), zscores.rename("zscore"), positions, weights, returns], axis=1
    )


# split the dates into rolling formation/trading windows. Pairs are selected on each formation
# period and traded over the trading period right after it. The windows step forward by the
# trading length, so trading periods are back to back and never overlap. Windows whose trading
# starts on or before validation_end are tagged validation, and later ones holdout.
def walk_forward_windows(
    dates: pd.DatetimeIndex,
    formation_months: int = 12,
    trading_months: int = 6,
    validation_end: str | None = None,
) -> pd.DataFrame:
    windows = []
    # Start the first formation period at the earliest available date.
    formation_start = dates[0]
    while True:
        # Formation ends after formation_months; trading begins immediately afterward.
        trading_start = formation_start + pd.DateOffset(months=formation_months)

        trading_end = trading_start + pd.DateOffset(months=trading_months)
        # Select actual dates within each period; < excludes the next period's start.
        formation_dates = dates[(dates >= formation_start) & (dates < trading_start)]
        trading_dates = dates[(dates >= trading_start) & (dates < trading_end)]
        if len(trading_dates) == 0:
            break
        windows.append({
            "formation_start": formation_dates[0],
            "formation_end": formation_dates[-1],
            "trading_start": trading_dates[0],
            "trading_end": trading_dates[-1],
        })

        # Advance by the trading length: formation periods overlap, trading periods don't.
        formation_start += pd.DateOffset(months=trading_months)
    windows = pd.DataFrame(windows)
    # If a cutoff exists, mark windows starting on/before it as validation.
    # Without a cutoff, treat every window as validation.
    if validation_end is not None:
        is_validation = (
            windows["trading_start"] <= pd.Timestamp(validation_end)
        )
    else:
        is_validation = True
    # Choose "validation" if True, otherwise "holdout", and assign to the period column.
    windows["period"] = np.where(is_validation, "validation", "holdout")
    return windows


# backtest every traded pair over its window's trading period, using the hedge ratio and spread
# std fit on that window's formation period. Kept separate from pair selection so the same pairs
# can be re-run with different signal settings.
def backtest_traded_pairs(
    log_prices: pd.DataFrame,
    scored_pairs: pd.DataFrame,
    windows: pd.DataFrame,
    signal_params: dict | None = None,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    pair_results = []
    for pair, row in scored_pairs[scored_pairs["traded"]].iterrows():
        window = windows.loc[row["window"]]
        result = backtest_pair(
            log_prices.loc[:window.trading_end], row["y"], row["x"], row["alpha"], row["beta"],
            row["spread_std"], **(signal_params or {}), cost_bps=cost_bps,
            start=window.trading_start, end=window.trading_end,
        )
        # Tag each daily result with its pair, window, and validation/holdout period.
        pair_results.append(result.assign(window=row["window"], period=window.period, pair=pair))
    if not pair_results:
        return pd.DataFrame(columns=["date", "window", "period", "pair", "net_return"])
    return pd.concat(pair_results).rename_axis("date").reset_index()


# select pairs on each formation period and backtest them over the following trading period.
# Returns the daily results of every traded pair and every pair scored in each window, with a
# traded column marking the ones that were selected and kept by max_pairs.
def walk_forward(
    log_prices: pd.DataFrame,
    universe: dict[str, list[str]],
    formation_months: int = 12,
    trading_months: int = 6,
    max_pairs: int | None = None,
    validation_end: str | None = None,
    pair_params: dict | None = None,
    signal_params: dict | None = None,
    cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = walk_forward_windows(
        log_prices.index, formation_months, trading_months, validation_end
    )
    scored_pairs = []
    for window_id, window in windows.iterrows():
        # pairs and hedge ratios only see formation data, so the trading period is out of
        # sample.
        formation = log_prices.loc[window.formation_start:window.formation_end]
        scored = select_pairs(formation, universe, **(pair_params or {}))
        # scored is sorted by p-value, so head keeps the strongest selected pairs, max_pairs if
        # specified.
        chosen = scored[scored["selected"]].head(max_pairs)
        scored_pairs.append(scored.assign(
            window=window_id, period=window.period, traded=scored.index.isin(chosen.index)
        ))
    scored_pairs = pd.concat(scored_pairs) if scored_pairs else pd.DataFrame()
    pair_results = backtest_traded_pairs(
        log_prices, scored_pairs, windows, signal_params, cost_bps
    )
    return pair_results, scored_pairs


if __name__ == "__main__":
    from src.data_loader import load_config, load_data, load_universe, to_log_prices
    from src.portfolio import equal_weight_returns

    cfg = load_config()
    wf_cfg = cfg["walk_forward"]
    prices, _, _ = load_data(cfg)
    log_prices = to_log_prices(prices)
    pair_results, scored_pairs = walk_forward(
        log_prices, load_universe(cfg), **wf_cfg,
        pair_params=cfg["pairs"], signal_params=cfg["signals"], **cfg["backtest"],
    )
    windows = walk_forward_windows(
        log_prices.index, wf_cfg["formation_months"], wf_cfg["trading_months"],
        wf_cfg["validation_end"],
    )
    # Count selected pairs per window; use an empty Series if none were selected.
    selections = scored_pairs[scored_pairs["traded"]] if len(scored_pairs) else scored_pairs
    counts = selections.groupby("window").size() if len(selections) else pd.Series(dtype=int)
    # Match counts to all windows, filling windows with no selected pairs with zero.
    windows["n_selected"] = counts.reindex(windows.index, fill_value=0)

    # only validation results are shown so tuning never sees the holdout.
    validation = windows[windows["period"] == "validation"]
    print(validation.to_string())
    trading_dates = log_prices.loc[
        validation["trading_start"].iloc[0]:validation["trading_end"].iloc[-1]
    ].index
    # Average selected pair returns into one portfolio return per trading day.
    portfolio = equal_weight_returns(
        pair_results[pair_results["period"] == "validation"], trading_dates
    )
    # Compound daily net returns into total validation return; display as a percentage.
    print(f"\nValidation cumulative net return: {(1 + portfolio).prod() - 1:.2%}")

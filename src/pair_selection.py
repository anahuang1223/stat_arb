from itertools import combinations

import pandas as pd

from src.statistical_tests import benjamini_hochberg, engle_granger, half_life, hedge_ratio


# build every within-sector pair of tickers that have price data. Pairs only come from the same
# sector since stocks with shared economic drivers are more likely to be cointegrated.
def candidate_pairs(
    universe: dict[str, list[str]], available: list[str]
) -> list[tuple[str, str, str]]:
    pairs = []
    for sector, tickers in universe.items():
        # drop tickers missing from the price data and sort so the Y/X order of each pair is
        # deterministic.
        kept = sorted(ticker for ticker in tickers if ticker in available)
        # each unordered pair is tested once with the alphabetically first ticker as Y.
        for y_ticker, x_ticker in combinations(kept, 2):
            pairs.append((sector, y_ticker, x_ticker))
    return pairs


# compute the hedge ratio, half-life, and Engle-Granger pval for every candidate pair.
def score_pairs(log_prices: pd.DataFrame, universe: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for sector, y_ticker, x_ticker in candidate_pairs(universe, list(log_prices.columns)):
        y_series, x_series = log_prices[y_ticker], log_prices[x_ticker]
        alpha, beta = hedge_ratio(y_series, x_series)
        # spread is the regression residual, which should be stationary if Y and X are
        # cointegrated.
        spread = y_series - alpha - beta * x_series
        rows.append({
            "pair": f"{y_ticker}/{x_ticker}",
            "sector": sector,
            "y": y_ticker,
            "x": x_ticker,
            "alpha": alpha,
            "beta": beta,
            "half_life": half_life(spread),
            "pvalue": engle_granger(y_series, x_series),
        })
    # return an empty frame with the expected columns so downstream code does not break when
    # no pairs are available.
    if not rows:
        return pd.DataFrame(
            columns=["sector", "y", "x", "alpha", "beta", "half_life", "pvalue"]
        )
    return pd.DataFrame(rows).set_index("pair")


# keep pairs that are cointegrated after BH correction and have a tradeable spread.
def select_pairs(
    log_prices: pd.DataFrame,
    universe: dict[str, list[str]],
    fdr_alpha: float = 0.05,
    min_half_life: float = 1,
    max_half_life: float = 60,
) -> pd.DataFrame:
    scored = score_pairs(log_prices, universe)
    # correct for testing many pairs at once so the expected share of false discoveries among
    # the significant pairs stays at or below fdr_alpha.
    bh = benjamini_hochberg(scored["pvalue"], alpha=fdr_alpha)
    scored["pvalue_adj"] = bh["pvalue_adj"]
    scored["significant"] = bh["significant"].astype(bool)
    # a pair is selected only if it is significant, has a positive hedge ratio (long one leg,
    # short the other), and reverts fast enough to trade but not so fast that it is likely
    # noise. An inf or NaN half-life fails the between check, so non-reverting spreads are
    # dropped.
    scored["selected"] = (
        scored["significant"]
        & (scored["beta"] > 0)
        & scored["half_life"].between(min_half_life, max_half_life)
    )
    return scored.sort_values("pvalue")


if __name__ == "__main__":
    from src.data_loader import load_config, load_data, load_universe

    cfg = load_config()
    # select pairs on the training set only to avoid look-ahead bias in the backtest.
    _, log_train, _ = load_data(cfg)
    result = select_pairs(log_train, load_universe(cfg), **cfg["pairs"])
    print(f"{len(result)} pairs tested, {result['significant'].sum()} significant after BH, "
          f"{result['selected'].sum()} selected")
    print(result.head(15).round(4).to_string())

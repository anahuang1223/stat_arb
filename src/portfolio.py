import pandas as pd


# combine the pairs into one daily return series with equal capital in every pair selected for
# that window. A pair that is flat still holds its share of capital and earns 0, and days with no
# selected pairs earn 0. return_column picks net or gross returns.
def equal_weight_returns(
    pair_results: pd.DataFrame, dates: pd.DatetimeIndex, return_column: str = "net_return"
) -> pd.Series:
    if pair_results.empty:
        return pd.Series(0.0, index=dates, name="portfolio_return")
    by_pair = pair_results.pivot_table(index="date", columns="pair", values=return_column)
    # a pair not selected in a window has NaN on those days, and mean skips NaN, so the average
    # only covers the pairs selected for that window.
    daily = by_pair.mean(axis=1)
    return daily.reindex(dates, fill_value=0.0).rename("portfolio_return")

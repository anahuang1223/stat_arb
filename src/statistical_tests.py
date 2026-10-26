import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def benjamini_hochberg(pvalues: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    pvals = pvalues.to_numpy(dtype=float)
    n_tests = len(pvals)
    if n_tests == 0:
        return pd.DataFrame(columns=["pvalue", "pvalue_adj", "significant"])
    if np.isnan(pvals).any() or ((pvals < 0) | (pvals > 1)).any():
        raise ValueError("p-values must be in [0, 1] with no NaNs")

    # array of indices that would sort the array from lowest to highest p.
    sort_order = np.argsort(pvals)
    # array of ranks 1 to n_tests.
    ranks = np.arange(1, n_tests + 1)
    # array of scaled pvals sorted by rank (lowest original p is rank 1) using the BH formula:
    # pvals * n_tests / rank.
    scaled_pvals = pvals[sort_order] * n_tests / ranks

    # H0 assumes the stock pair is not cointegrated; smaller pvals mean stronger evidence
    # against H0. If a higher rank (weaker evidence) passes its BH cutoff, all lower ranks
    # (stronger evidence) are also selected by the BH rule even if the lower ranks do not pass
    # their own cutoff. This is equivalent to taking the running minimum of the scaled pvals
    # from highest to lowest original rank with the lower ranks inheriting a higher rank's
    # smaller scaled value.
    adj_sorted = np.minimum.accumulate(scaled_pvals[::-1])[::-1]

    adj_pvals = np.empty(n_tests)
    # Restore adjusted pvals to their original order. Each index i in adj_pvals
    # corresponds to one stock pair (one cointegration test).
    adj_pvals[sort_order] = np.minimum(adj_sorted, 1.0)

    return pd.DataFrame(
        {
            "pvalue": pvals,
            "pvalue_adj": adj_pvals,
            # check which pairs pass the bh threshold, if the adj pval <= alpha then it is
            # statistically significant.
            "significant": adj_pvals <= alpha,
        },
        index=pvalues.index,
    )


# fit Y = alpha + beta * X where beta is the slope (hedge ratio) and alpha is the intercept.
# return both so the spread can be built as the regression residual: Y - alpha - beta * X.
def hedge_ratio(y_series: pd.Series, x_series: pd.Series) -> tuple[float, float]:
    beta, alpha = np.polyfit(x_series.to_numpy(dtype=float), y_series.to_numpy(dtype=float), 1)
    return float(alpha), float(beta)


# estimate how quickly the spread returns toward its equilibrium (mean).
def half_life(spread: pd.Series) -> float:
    spread_vals = spread.to_numpy(dtype=float)
    # array of previous spread values (lagged) paired with array of their changes in the next
    # period (diffs).
    lagged, diffs = spread_vals[:-1], np.diff(spread_vals)
    # slope of diffs on lagged, which measures how much the next change moves for each 1-unit
    # increase in the previous spread. A negative slope means a wide spread tends to shrink back
    # toward its mean (mean reversion).
    mean_rev_coef = np.polyfit(lagged, diffs, 1)[0]
    # a nonnegative slope means the spread does not mean revert, so the half-life is infinite.
    if mean_rev_coef >= 0:
        return np.inf
    # a slope of -1 or below means the spread fully resets or overshoots every period, so
    # log(1 + mean_rev_coef) is undefined and there is no valid half-life.
    if mean_rev_coef <= -1:
        return np.nan
    # number of periods for a deviation to shrink by half using the half-life formula:
    # -ln(2) / ln(1 + mean_rev_coef).
    return float(-np.log(2) / np.log(1 + mean_rev_coef))


# Engle-Granger test where H0 is that Y and X are not cointegrated.
def engle_granger(y_series: pd.Series, x_series: pd.Series) -> float:
    _, pvalue, _ = coint(y_series.to_numpy(dtype=float), x_series.to_numpy(dtype=float), trend="c")
    return float(pvalue)

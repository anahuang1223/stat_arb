import numpy as np
import pandas as pd


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

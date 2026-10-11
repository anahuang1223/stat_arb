import numpy as np
import pandas as pd


def benjamini_hochberg(pvalues: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    raw_pvalues = pvalues.to_numpy(dtype=float)
    n_tests = len(raw_pvalues)
    if n_tests == 0:
        return pd.DataFrame(columns=["pvalue", "pvalue_adj", "significant"])
    if np.isnan(raw_pvalues).any() or ((raw_pvalues < 0) | (raw_pvalues > 1)).any():
        raise ValueError("p-values must be in [0, 1] with no NaNs")

    sort_order = np.argsort(raw_pvalues)
    ranks = np.arange(1, n_tests + 1)
    scaled_pvalues = raw_pvalues[sort_order] * n_tests / ranks

    adj_sorted = np.minimum.accumulate(scaled_pvalues[::-1])[::-1]
    adj_pvalues = np.empty(n_tests)
    adj_pvalues[sort_order] = np.minimum(adj_sorted, 1.0)

    return pd.DataFrame(
        {
            "pvalue": raw_pvalues,
            "pvalue_adj": adj_pvalues,
            "significant": adj_pvalues <= alpha,
        },
        index=pvalues.index,
    )

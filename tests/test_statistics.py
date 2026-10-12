import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from src.statistical_tests import benjamini_hochberg


# Check that BH results match the statsmodels implementation.
def test_bh_vs_statsmodels():
    rng = np.random.default_rng(0)
    p = pd.Series(rng.uniform(0, 0.2, 40))
    ours = benjamini_hochberg(p, alpha=0.05)
    reject, adj, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    np.testing.assert_allclose(ours["pvalue_adj"], adj)
    np.testing.assert_array_equal(ours["significant"], reject)


# Check that BH produces the expected significance decisions and adj pval.
def test_bh_hand_example():
    p = pd.Series([0.04, 0.001, 0.5, 0.03, 0.01], index=list("abcde"))
    out = benjamini_hochberg(p, alpha=0.05)
    assert out["significant"].to_dict() == {
        "a": True, "b": True, "c": False, "d": True, "e": True,
    }
    assert out.loc["a", "pvalue_adj"] == pytest.approx(0.05)


# Check that BH accounts for the number of tests and is stricter than the raw pvals.
def test_bh_stricter_than_raw():
    out = benjamini_hochberg(pd.Series([0.04] + [0.9] * 19), alpha=0.05)
    assert out.loc[0, "pvalue_adj"] == pytest.approx(0.8)
    assert not out["significant"].any()


# Check that all pairs remain significant when every pval is sufficiently small.
def test_bh_many_small():
    out = benjamini_hochberg(pd.Series([0.04] * 20), alpha=0.05)
    assert out["significant"].all()


# Check that invalid pvals (NaN) raise a ValueError.
def test_bh_invalid():
    with pytest.raises(ValueError):
        benjamini_hochberg(pd.Series([0.1, np.nan]))


# Check that an empty input returns an empty DataFrame.
def test_bh_empty():
    assert benjamini_hochberg(pd.Series([], dtype=float)).empty

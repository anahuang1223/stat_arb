import numpy as np
import pandas as pd
import pytest

from src.signals import generate_positions, rolling_zscore


# Check that the z-score is NaN during warm-up and matches a hand-computed value after.
def test_rolling_zscore_values():
    spread = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    zscores = rolling_zscore(spread, window=3)
    assert zscores.iloc[:2].isna().all()
    # window [1, 2, 3] has mean 2 and std 1, so (3 - 2) / 1 = 1.
    assert zscores.iloc[2] == pytest.approx(1.0)


# Check that a flat spread gives NaN instead of dividing by zero.
def test_rolling_zscore_flat_spread():
    zscores = rolling_zscore(pd.Series([1.0, 1.0, 1.0, 1.0]), window=2)
    assert zscores.isna().all()


# Check entries, holds, exits, and stop-losses follow the expected positions.
def test_generate_positions_path():
    zscores = pd.Series([np.nan, 0.0, 2.1, 1.0, 0.4, -2.5, -1.0, 0.0, 5.0, 3.0, 4.5])
    expected = [0, 0, -1, -1, 0, 1, 1, 0, 0, -1, 0]
    assert generate_positions(zscores).tolist() == expected


# Check that a existing long exits instead of immediately reversing to short.
def test_generate_positions_no_flip():
    zscores = pd.Series([-2.5, 2.5])
    assert generate_positions(zscores).tolist() == [1, 0]


# Check that invalid entry/exit thresholds raise a ValueError.
def test_generate_positions_bad_thresholds():
    with pytest.raises(ValueError):
        generate_positions(pd.Series([0.0]), entry_z=1.0, exit_z=2.0)

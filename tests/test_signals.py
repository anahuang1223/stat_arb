import numpy as np
import pandas as pd
import pytest

from src.signals import formation_zscore, generate_positions


# Check that the z-score is the spread in units of the formation std.
def test_formation_zscore_values():
    zscores = formation_zscore(pd.Series([0.0, 0.2, -0.1]), spread_std=0.1)
    assert zscores.tolist() == pytest.approx([0.0, 2.0, -1.0])


# Check that a spread drifting away keeps a growing z-score instead of reverting, which a rolling
# z-score would hide as its mean and std follow the drift.
def test_formation_zscore_drift_stays_large():
    zscores = formation_zscore(pd.Series(np.linspace(0.0, 1.0, 50)), spread_std=0.1)
    assert zscores.is_monotonic_increasing
    assert zscores.iloc[-1] == pytest.approx(10.0)


# Check that a flat formation spread gives NaN instead of dividing by zero.
def test_formation_zscore_flat_spread():
    zscores = formation_zscore(pd.Series([1.0, 1.0]), spread_std=0.0)
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

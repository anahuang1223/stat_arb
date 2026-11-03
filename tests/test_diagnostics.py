import pandas as pd

from src.diagnostics import label_exits, screening_summary
from src.metrics import extract_trades


# Check that exits are labeled stop, reverted, or window_end from the exit-day z-score.
def test_label_exits():
    dates = pd.bdate_range("2020-01-01", periods=9)
    pair_results = pd.DataFrame({
        "date": dates,
        "window": 0,
        "period": "validation",
        "pair": "A/B",
        # a long that reverts, a short that stops out, and a long closed at the window end.
        "position": [1, 0, -1, -1, 0, 1, 1, 1, 0],
        "zscore": [-2.1, -0.3, 2.2, 3.0, 4.2, -2.5, -2.0, -1.5, -1.2],
        "net_return": 0.0,
    })
    trades = label_exits(extract_trades(pair_results), pair_results, exit_z=0.5, stop_z=4.0)
    assert trades["exit_reason"].tolist() == ["reverted", "stop", "window_end"]
    assert trades["exit_zscore"].tolist() == [-0.3, 4.2, -1.2]


# Check the per-window counts and that a stock shared by two traded pairs is counted once.
def test_screening_summary():
    windows = pd.DataFrame({"period": ["validation"], "trading_start": [pd.Timestamp("2020-01-01")]})
    scored_pairs = pd.DataFrame({
        "window": 0,
        "y": ["A", "A", "C", "D"],
        "x": ["B", "C", "D", "E"],
        "pvalue": [0.001, 0.002, 0.04, 0.5],
        "significant": [True, True, False, False],
        "selected": [True, True, False, False],
        "traded": [True, True, False, False],
    }, index=["A/B", "A/C", "C/D", "D/E"])
    row = screening_summary(scored_pairs, windows).iloc[0]
    assert row["n_tested"] == 4
    assert row["n_raw_p05"] == 3
    assert row["n_expected_by_chance"] == 0.2
    assert row["n_traded"] == 2
    assert row["n_distinct_stocks"] == 3
    assert row["max_pairs_per_stock"] == 2

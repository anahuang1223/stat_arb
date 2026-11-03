import numpy as np
import pandas as pd


# Use the standard deviation from the formation period to calculate each day's z-score.
# The spread's average during formation is about 0, so we divide the spread by that std.
# Unlike a rolling z-score, this std stays fixed during trading.
# If the stocks keep moving apart, the z-score keeps growing and can hit our stop-loss,
# instead of appearing to return to normal because the rolling average moved.
def formation_zscore(spread: pd.Series, spread_std: float) -> pd.Series:
    # a flat formation spread has zero std, so the z-score is undefined.
    if not spread_std > 0:
        return pd.Series(np.nan, index=spread.index)
    return spread / spread_std


# turn z-scores into positions on the spread: +1 is long the spread (long Y, short beta * X),
# -1 is short the spread, and 0 is flat. A position is decided on day t's close, so the
# backtest must shift it by one day before applying returns.
# after_stop sets what happens once a trade is stopped out: "reenter" allows a new entry as soon
# as the z-score is back inside the entry band, "reset" waits until the spread has come back
# within exit_z of its mean, and "retire" never trades the pair again in this series.
def generate_positions(
    zscores: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    after_stop: str = "reenter",
) -> pd.Series:
    if not 0 <= exit_z < entry_z < stop_z:
        raise ValueError("thresholds must satisfy 0 <= exit_z < entry_z < stop_z")
    if after_stop not in ("reenter", "reset", "retire"):
        raise ValueError("after_stop must be 'reenter', 'reset', or 'retire'")

    positions = np.zeros(len(zscores))
    current = 0
    # True after a stop while new entries are not allowed.
    blocked = False
    for idx, z_value in enumerate(zscores.to_numpy(dtype=float)):
        # no z-score (warm-up window or flat spread) means stay flat.
        if np.isnan(z_value):
            current = 0
        elif current == 0:
            if blocked and after_stop == "reset" and abs(z_value) <= exit_z:
                blocked = False
            # enter when the spread is stretched but not so far that the pair looks broken.
            # A spread far above its mean is expected to fall, so short it, and vice versa.
            if not blocked and entry_z <= z_value < stop_z:
                current = -1
            elif not blocked and -stop_z < z_value <= -entry_z:
                current = 1
        # stop out of a long if the spread keeps falling, or exit once it reverts near its mean.
        elif current == 1:
            if z_value <= -stop_z:
                current = 0
                blocked = after_stop != "reenter"
            elif z_value >= -exit_z:
                current = 0
        # stop out of a short if the spread keeps rising, or exit once it reverts near its mean.
        elif current == -1:
            if z_value >= stop_z:
                current = 0
                blocked = after_stop != "reenter"
            elif z_value <= exit_z:
                current = 0
        positions[idx] = current
    return pd.Series(positions, index=zscores.index, name="position")

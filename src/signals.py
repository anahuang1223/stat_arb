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
def generate_positions(
    zscores: pd.Series, entry_z: float = 2.0, exit_z: float = 0.5, stop_z: float = 4.0
) -> pd.Series:
    if not 0 <= exit_z < entry_z < stop_z:
        raise ValueError("thresholds must satisfy 0 <= exit_z < entry_z < stop_z")

    positions = np.zeros(len(zscores))
    current = 0
    for idx, z_value in enumerate(zscores.to_numpy(dtype=float)):
        # no z-score (warm-up window or flat spread) means stay flat.
        if np.isnan(z_value):
            current = 0
        elif current == 0:
            # enter when the spread is stretched but not so far that the pair looks broken.
            # A spread far above its mean is expected to fall, so short it, and vice versa.
            if entry_z <= z_value < stop_z:
                current = -1
            elif -stop_z < z_value <= -entry_z:
                current = 1
        # exit a long once the spread reverts up near its mean, or stop out if it keeps falling.
        elif current == 1 and (z_value >= -exit_z or z_value <= -stop_z):
            current = 0
        # exit a short once the spread reverts down near its mean, or stop out if it keeps rising.
        elif current == -1 and (z_value <= exit_z or z_value >= stop_z):
            current = 0
        positions[idx] = current
    return pd.Series(positions, index=zscores.index, name="position")

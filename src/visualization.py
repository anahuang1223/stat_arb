from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
FORMATION_SHADE = "#f0efec"
SERIES_COLORS = ["#2a78d6", "#eb6834"]
STOP_COLOR = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_MUTED,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "font.size": 9,
    "legend.frameon": False,
})


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# histogram of every Engle-Granger p-value tested across all windows. If no pair were truly
# cointegrated the p-values would be uniform, so each 0.05-wide bin would hold about 1/20 of the
# tests; the dashed line marks that level.
def plot_pvalue_histogram(pvalues: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = [i / 20 for i in range(21)]
    ax.hist(pvalues, bins=bins, color=SERIES_COLORS[0], edgecolor=SURFACE, linewidth=1.5)
    expected = len(pvalues) / 20
    ax.axhline(expected, color=TEXT_MUTED, linestyle="--", linewidth=1)
    ax.annotate("expected if no pair were cointegrated", (1.0, expected),
                xytext=(0, 4), textcoords="offset points", ha="right", color=TEXT_MUTED)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Engle-Granger p-value")
    ax.set_ylabel("pairs tested")
    ax.set_title(f"P-values of {len(pvalues):,} pair tests across all formation windows",
                 loc="left")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


# cumulative return of the portfolio in each period, net and gross of costs, one panel per period
# on a shared y-axis so the two periods can be compared directly.
def plot_equity_curves(curves: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, len(curves), figsize=(10, 3.5), sharey=True, squeeze=False)
    for ax, (period, returns) in zip(axes[0], curves.items()):
        for color, column in zip(SERIES_COLORS, ["net", "gross"]):
            cumulative = (1 + returns[column]).cumprod() - 1
            ax.plot(cumulative.index, cumulative * 100, color=color, linewidth=1.5, label=column)
            ax.annotate(f"{column} {cumulative.iloc[-1]:+.1%}", (cumulative.index[-1],
                        cumulative.iloc[-1] * 100), xytext=(4, 0), textcoords="offset points",
                        va="center", color=TEXT)
        ax.axhline(0, color=TEXT_MUTED, linewidth=0.8)
        ax.set_title(period.capitalize(), loc="left")
    axes[0][0].set_ylabel("cumulative return (%)")
    axes[0][0].legend(loc="upper left")
    fig.autofmt_xdate()
    _save(fig, path)


# formation-period z-score of a few pairs' spreads through formation and trading. The shaded
# region is formation, where the hedge ratio and spread std were fit; the bands are the entry and
# stop thresholds the trading period is judged against.
def plot_spread_examples(
    examples: list[dict], entry_z: float, exit_z: float, stop_z: float, path: Path
) -> None:
    fig, axes = plt.subplots(len(examples), 1, figsize=(9, 2.6 * len(examples)), sharey=True,
                             squeeze=False)
    for ax, example in zip(axes[:, 0], examples):
        zscores = example["zscores"]
        ax.axvspan(zscores.index[0], example["trading_start"], color=FORMATION_SHADE, zorder=0)
        ax.plot(zscores.index, zscores, color=SERIES_COLORS[0], linewidth=1)
        for level in (entry_z, -entry_z):
            ax.axhline(level, color=TEXT_MUTED, linestyle="--", linewidth=0.8)
        for level in (exit_z, -exit_z):
            ax.axhline(level, color=TEXT_MUTED, linestyle=":", linewidth=0.8)
        for level in (stop_z, -stop_z):
            ax.axhline(level, color=STOP_COLOR, linestyle="--", linewidth=0.8)
        ax.axhline(0, color=TEXT_MUTED, linewidth=0.6)
        ax.set_title(example["title"], loc="left")
        ax.set_ylabel("z-score")
    axes[0, 0].annotate("formation", (0.01, 0.92), xycoords="axes fraction", color=TEXT_MUTED)
    axes[0, 0].annotate(f"entry ±{entry_z:g}  exit ±{exit_z:g}  stop ±{stop_z:g}",
                        (0.99, 0.92), xycoords="axes fraction", ha="right", color=TEXT_MUTED)
    fig.tight_layout()
    _save(fig, path)

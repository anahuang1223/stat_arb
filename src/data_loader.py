from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = PROJECT_ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def all_tickers(universe: dict[str, list[str]]) -> list[str]:
    return sorted({t for tickers in universe.values() for t in tickers})


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path,
    force: bool = False,
) -> pd.DataFrame:
    cache_dir = PROJECT_ROOT / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"prices_{start}_{end}.csv"

    if cache_file.exists() and not force:
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if set(tickers) <= set(cached.columns):
            return cached[tickers]

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    prices.to_csv(cache_file)
    return prices[tickers]


def clean_prices(prices: pd.DataFrame, max_missing_frac: float = 0.05) -> pd.DataFrame:
    missing = prices.isna().mean()
    dropped = missing[missing > max_missing_frac].index.tolist()
    if dropped:
        print(f"Dropping tickers with >{max_missing_frac:.0%} missing data: {dropped}")
    prices = prices.drop(columns=dropped)

    prices = prices.ffill().dropna()
    return prices[(prices > 0).all(axis=1)]


def to_log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices)


def train_test_split(df: pd.DataFrame, train_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[:train_end]
    test = df.loc[df.index > pd.Timestamp(train_end)]
    return train, test


def load_data(config: dict | None = None, force_download: bool = False):
    cfg = (config or load_config())["data"]
    tickers = all_tickers(cfg["universe"])

    prices = download_prices(
        tickers, cfg["start_date"], cfg["end_date"], cfg["raw_dir"], force=force_download
    )
    prices = clean_prices(prices, cfg["max_missing_frac"])

    processed = PROJECT_ROOT / cfg["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)
    prices.to_csv(processed / "prices_clean.csv")

    log_train, log_test = train_test_split(to_log_prices(prices), cfg["train_end"])
    return prices, log_train, log_test


def missing_data_sensitivity(
    raw_prices: pd.DataFrame, thresholds: list[float]
) -> pd.DataFrame:
    missing = raw_prices.isna().mean()
    rows = []
    for t in sorted(thresholds):
        kept = missing[missing <= t].index
        rows.append({
            "threshold": t,
            "n_kept": len(kept),
            "dropped": sorted(set(raw_prices.columns) - set(kept)),
        })
    return pd.DataFrame(rows).set_index("threshold")


if __name__ == "__main__":
    prices, train, test = load_data()
    print(f"{prices.shape[1]} tickers, {len(prices)} trading days")
    print(f"Train: {train.index[0].date()} -> {train.index[-1].date()} ({len(train)} days)")
    print(f"Test:  {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} days)")
    print(prices.tail(3).round(2))

    cfg = load_config()
    raw = download_prices(
        all_tickers(cfg["data"]["universe"]),
        cfg["data"]["start_date"], cfg["data"]["end_date"], cfg["data"]["raw_dir"],
    )
    print("\nMissing-data sensitivity:")
    print(missing_data_sensitivity(raw, cfg["robustness"]["missing_frac_grid"]))

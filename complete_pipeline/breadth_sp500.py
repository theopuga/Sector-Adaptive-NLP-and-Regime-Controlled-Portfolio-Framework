import os
import math
import time
import io
import time
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Iterable
import requests

START = "2005-01-01"
END   = None
BATCH = 60
DELAY = 1.0

OUT_DAILY  = "./data/spx_breadth_daily.csv"
OUT_MONTHLY= "./data/spx_breadth_monthly.csv"

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def _normalize_symbols(symbols):
    # Yahoo uses '-' instead of '.' (e.g., BRK.B -> BRK-B)
    return sorted(set([s.replace(".", "-").strip().upper() for s in symbols if isinstance(s, str) and s.strip()]))

def chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def get_sp500_tickers():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }
    try:
        r = requests.get(WIKI_URL, headers=headers, timeout=20)
        r.raise_for_status()
        # parse the FIRST table on the page
        tables = pd.read_html(io.StringIO(r.text))
        df = tables[0]
        if "Symbol" in df.columns:
            syms = _normalize_symbols(df["Symbol"].tolist())
            return syms
    except Exception as e:
        print(f"[WARN] Wikipedia fetch failed: {e}")

    raise RuntimeError("Unable to obtain S&P 500 tickers (Wikipedia blocked, fallback failed, no cache).")

def download_batch(tickers: List[str], start: str, end: str | None):
    df = yf.download(tickers, start="2000-01-01", end=end, auto_adjust=True, progress=False, group_by="ticker", threads=True, interval="1d")
    # Normalize to a single wide DataFrame of Adj Close
    # yfinance returns multi-index columns when multiple tickers
    print("df.columns: ", df.columns)
    if isinstance(df.columns, pd.MultiIndex):
        # prefer 'Adj Close' if present; otherwise 'Close'
        fields = set([lvl[1] for lvl in df.columns])
        col = "Adj Close" if "Adj Close" in fields else "Close"
        out = {t: df[(t, col)] for t in tickers if (t, col) in df.columns}
        wide = pd.DataFrame(out)
    else:
        # single ticker case
        wide = df["Adj Close"].to_frame()
        wide.columns = tickers[:1]
    return wide

def download_prices_yf(tickers: List[str], start: str, end: str | None, batch=BATCH, delay=DELAY) -> pd.DataFrame:
    frames = []
    for b, group in enumerate(chunked(tickers, batch), 1):
        try:
            w = download_batch(group, start, end)
            frames.append(w)
        except Exception as e:
            print(f"[WARN] batch {b} failed ({len(group)} tickers): {e}")
            raise e
        time.sleep(delay)
    if not frames:
        raise RuntimeError("No data downloaded.")
    # Outer-join on date to keep as many names as possible
    data = pd.concat(frames, axis=1).sort_index()
    # Drop columns that are completely empty
    data = data.dropna(axis=1, how="all")
    return data

def compute_breadth_pct_above_ma(data: pd.DataFrame, ma_window=200) -> pd.Series:
    ma = data.rolling(ma_window, min_periods=ma_window).mean()
    # Compare price vs MA (require both non-null)
    valid = (~data.isna()) & (~ma.isna())
    above = (data > ma) & valid
    breadth = above.sum(axis=1) / valid.sum(axis=1)
    breadth.name = f"pct_above_{ma_window}d_ma"
    return breadth

def to_month_end(breadth_daily: pd.Series) -> pd.Series:
    # Use month-end last value (no lookahead within month)
    monthly = breadth_daily.resample("M").last()
    monthly.name = breadth_daily.name
    return monthly

def main():
    print("[INFO] Fetching current S&P500 tickers…")
    tickers = get_sp500_tickers()
    print(f"[INFO] Got {len(tickers)} tickers.")

    print("[INFO] Downloading daily prices from Yahoo…")
    prices = download_prices_yf(tickers, START, END)
    print(f"[INFO] Price frame: {prices.shape[0]} days × {prices.shape[1]} tickers")

    print("[INFO] Computing 200-day breadth…")
    breadth_daily = compute_breadth_pct_above_ma(prices, ma_window=200)
    breadth_monthly = to_month_end(breadth_daily)

    # Save
    breadth_daily.to_csv(OUT_DAILY, header=True)
    breadth_monthly.to_csv(OUT_MONTHLY, header=True)
    print(f"[OK] Saved daily breadth → {OUT_DAILY}")
    print(f"[OK] Saved monthly breadth → {OUT_MONTHLY}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Make Sector Momentum
====================
Purpose:
    Aggregates word-level frequencies into sector-level series and computes  
    **sector momentum metrics** over time (1M, 3M, 6M, 12M).  
    Produces sector-level summaries used for backtesting or thematic analysis.

Inputs:
    --panel : Parquet file with (gram, date, tf_share/df_share)
    --sector_dict_in : Validated or seed sector dictionary JSON
    --first_trading_year : Earliest year to include in the freeze
    --use_share : Metric to use ('tf_share' or 'df_share')
    --out_dir : Output directory for processed sector data

Outputs:
    1. `sector_word_frequency_monthly.parquet` — monthly aggregate word frequencies per sector  
    2. `sector_word_momentum_monthly.parquet` — includes 1M–12M momentum features  
    3. `sector_momentum_summary.csv` — top 3 sectors per month by 6M momentum  
    4. Frozen dictionary JSON (leak-free) written to `sector_dict_out`

Key Features:
    - Validates that each word existed before the cutoff year.
    - Allocates term shares across sectors proportionally when words belong to multiple sectors.
    - Computes momentum ratios for trend analysis.

Command:
python run/make_sector_momentum.py \
  --panel data/textsec/processed/word_trends_monthly.parquet \
  --sector_dict_in data/textsec/processed/FreezeDict/sector_dictionary_seed.json \
  --sector_dict_out data/textsec/processed/FreezeDict/sector_dictionary.json \
  --first_trading_year 2015 \
  --use_share tf_share \
  --out_dir data/textsec/processed/StaticDict
"""

import json, argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict

def load_panel(parquet_fp: str | Path) -> pd.DataFrame:
    """Load panel, optimized for large files by reading only needed columns."""
    parquet_fp = Path(parquet_fp)
    file_size_mb = parquet_fp.stat().st_size / (1024 * 1024)
    
    # For large files, use column selection
    cols_needed = ["date", "gram", "tf_share", "df_share"]
    
    if file_size_mb > 5000:
        try:
            import polars as pl
            print(f"[LOAD] Large file ({file_size_mb:.1f} MB), using Polars...")
            df = pl.read_parquet(parquet_fp, columns=cols_needed).to_pandas()
        except ImportError:
            print(f"[LOAD] Large file ({file_size_mb:.1f} MB), using Pandas with column selection...")
            df = pd.read_parquet(parquet_fp, columns=cols_needed, engine='pyarrow')
    else:
        df = pd.read_parquet(parquet_fp)
        cols_needed_present = [c for c in cols_needed if c in df.columns]
        df = df[cols_needed_present]
    
    if "date" not in df.columns or "gram" not in df.columns:
        raise ValueError("Expected columns 'date' and 'gram' not found.")
    df["date"] = pd.to_datetime(df["date"])
    return df

def build_reverse_map(sector_dict: dict[str, list[str]]) -> dict[str, set[str]]:
    rev = defaultdict(set)
    for sector, words in sector_dict.items():
        for w in words:
            rev[str(w).strip().lower()].add(sector)
    return rev

def validate_terms_exist_before(df: pd.DataFrame, words: set[str], cutoff: pd.Timestamp) -> set[str]:
    seen = df[df["gram"].isin(words)].groupby("gram")["date"].min()
    ok = set(seen[seen < cutoff].index)
    return ok

def allocate_shares(df: pd.DataFrame, rev_map: dict[str, set[str]], use="tf_share") -> pd.DataFrame:
    if use not in df.columns:
        raise ValueError(f"Column '{use}' not in panel.")
    sub = df[df["gram"].isin(rev_map.keys())].copy()
    sub["sectors"] = sub["gram"].map(lambda g: sorted(rev_map[g]))
    sub = sub.explode("sectors")
    counts = sub.groupby(["date","gram"])["sectors"].transform("count")
    sub["alloc"] = sub[use] / counts
    out = (
        sub.groupby(["date","sectors"], as_index=False)["alloc"].sum()
        .rename(columns={"sectors":"sector", "alloc":f"sector_{use}"})
        .sort_values(["sector","date"])
    )
    return out

def add_sector_momentum(df: pd.DataFrame, use="tf_share") -> pd.DataFrame:
    col = f"sector_{use}"
    df = df.sort_values(["sector","date"]).copy()
    g = df.groupby("sector", sort=False)[col]
    def roll_ret(s, k): return s.divide(s.shift(k)) - 1.0
    df["mom_1m"] = roll_ret(g.transform(lambda x: x), 1)
    df["mom_3m"] = roll_ret(g.transform(lambda x: x), 3)
    df["mom_6m"] = roll_ret(g.transform(lambda x: x), 6)
    df["mom_12m"] = roll_ret(g.transform(lambda x: x), 12)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="data/textsec/processed/word_trends_monthly.parquet")
    ap.add_argument("--sector_dict_in", required=True, help="seed sector dict (JSON) to validate & freeze")
    ap.add_argument("--sector_dict_out", required=True, help="frozen validated dict (JSON)")
    ap.add_argument("--first_trading_year", type=int, default=2015)
    ap.add_argument("--use_share", choices=["tf_share","df_share"], default="tf_share")
    ap.add_argument("--out_dir", default="data/textsec/processed")
    args = ap.parse_args()

    panel = load_panel(args.panel)

    raw = json.loads(Path(args.sector_dict_in).read_text())
    sector_dict = {k: [str(w).lower() for w in v] for k, v in raw.items()}
    rev_map = build_reverse_map(sector_dict)

    cutoff = pd.Timestamp(f"{args.first_trading_year}-01-01")
    valid_words = validate_terms_exist_before(panel, set(rev_map.keys()), cutoff)

    frozen_rev = {w: rev_map[w] for w in valid_words}
    frozen_fwd = defaultdict(list)
    for w, secs in frozen_rev.items():
        for s in secs:
            frozen_fwd[s].append(w)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.sector_dict_out).write_text(json.dumps({k: sorted(set(v)) for k, v in frozen_fwd.items()}, indent=2))

    sector_freq = allocate_shares(panel, frozen_rev, use=args.use_share)
    sector_mom = add_sector_momentum(sector_freq, use=args.use_share)

    freq_fp = out_dir / "sector_word_frequency_monthly.parquet"
    mom_fp  = out_dir / "sector_word_momentum_monthly.parquet"
    sector_freq.to_parquet(freq_fp, index=False)
    sector_mom.to_parquet(mom_fp, index=False)

    summary = (
        sector_mom.dropna(subset=["mom_6m"])
        .assign(year=lambda d: d["date"].dt.year, month=lambda d: d["date"].dt.month)
        .sort_values(["date","mom_6m"], ascending=[True, False])
        .groupby("date").head(3)[["date","sector","mom_6m"]]
    )
    summary_fp = out_dir / "sector_momentum_summary.csv"
    summary.to_csv(summary_fp, index=False)

    print("Frozen dict written to:", args.sector_dict_out)
    print("Wrote:", freq_fp, mom_fp, summary_fp)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Freeze Sector Dictionary
=========================
Purpose:
    This script freezes an initial sector dictionary to prevent data leakage from future periods.  
    It validates each word’s appearance history in `word_trends_monthly.parquet`, ensuring that  
    only words observed *before* a given cutoff year (and meeting minimum pre-cutoff presence)  
    remain in the sector dictionary.

Inputs:
    --panel : Path to the word-trend Parquet file (e.g., data/textsec/processed/word_trends_monthly.parquet)
    --sector_dict_in : Seed sector dictionary JSON (raw, human-curated)
    --cutoff_year : Year before which words must appear to be included
    --strict_year : Optional stricter upper limit on first appearance
    --min_pre_cutoff_months : Minimum number of months word must appear before cutoff
    --min_pre_cutoff_doc_months : Minimum months with non-zero docs before cutoff

Outputs:
    1. Frozen sector dictionary JSON (leak-free)  
       → e.g. `data/textsec/processed/FreezeDict/sector_dictionary.json`
    2. Appearance report CSV (for audit & diagnostics)  
       → e.g. `data/textsec/processed/FreezeDict/sector_word_appearance_report.csv`

Key Features:
    - Ensures no look-ahead bias by restricting words to pre-cutoff history.
    - Provides detailed reason codes for excluded words.
    - Builds both forward (sector→words) and reverse (word→sectors) maps.

# Freeze initial dictionary command
python run/freeze_sector_dictionary.py \
  --panel data/textsec/processed/word_trends_monthly.parquet \
  --sector_dict_in resources/sector_dictionary_seed.json \
  --sector_dict_out data/textsec/processed/FreezeDict/sector_dictionary.json \
  --appearance_report_out data/textsec/processed/FreezeDict/sector_word_appearance_report.csv \
  --cutoff_year 2015 \
  --strict_year 2010 \
  --min_pre_cutoff_months 3 \
  --min_pre_cutoff_doc_months 2
"""
import json, argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict

def load_panel(parquet_fp: str | Path) -> pd.DataFrame:
    """Load panel, optimized for large files by reading only needed columns."""
    parquet_fp = Path(parquet_fp)
    file_size_mb = parquet_fp.stat().st_size / (1024 * 1024)
    
    # For large files, use column selection and Polars if available
    cols_needed = ["date", "gram", "tf_share", "df_share", "month_docs"]
    
    if file_size_mb > 5000:
        try:
            import polars as pl
            print(f"[LOAD] Large file ({file_size_mb:.1f} MB), using Polars...")
            # Read only needed columns with Polars
            df = pl.read_parquet(parquet_fp, columns=cols_needed).to_pandas()
        except ImportError:
            print(f"[LOAD] Large file ({file_size_mb:.1f} MB), using Pandas with column selection...")
            df = pd.read_parquet(parquet_fp, columns=cols_needed, engine='pyarrow')
    else:
        df = pd.read_parquet(parquet_fp)
        # Filter to needed columns if we read all
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

def compute_word_stats(df: pd.DataFrame, words: set[str], cutoff: pd.Timestamp) -> pd.DataFrame:
    sub = df[df["gram"].isin(words)].copy()
    sub_before = sub[sub["date"] < cutoff]
    first_seen = sub.groupby("gram")["date"].min().rename("first_seen")
    first_seen_before = sub_before.groupby("gram")["date"].min().rename("first_seen_before_cutoff")
    months_before = sub_before.groupby("gram")["date"].nunique().rename("months_before_cutoff")
    # if month_docs exists, count distinct months with any docs>0 (breadth)
    if "month_docs" in sub_before.columns:
        docs_before = (sub_before.assign(has_docs=lambda d: d["month_docs"] > 0)
                                 .groupby("gram")["has_docs"].sum()
                                 .rename("pre_cutoff_doc_months"))
    else:
        docs_before = pd.Series(dtype="float64", name="pre_cutoff_doc_months")

    stats = pd.concat([first_seen, first_seen_before, months_before, docs_before], axis=1)
    stats.index.name = "word"
    return stats.reset_index()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="data/textsec/processed/word_trends_monthly.parquet")
    ap.add_argument("--sector_dict_in", required=True, help="seed dictionary JSON (raw)")
    ap.add_argument("--sector_dict_out", required=True, help="frozen dictionary JSON (leak-free)")
    ap.add_argument("--appearance_report_out", required=True, help="CSV report of word first appearances")
    ap.add_argument("--cutoff_year", type=int, default=2015, help="No-lookahead cutoff year (default 2015)")
    ap.add_argument("--strict_year", type=int, default=None,
                    help="Optional stricter year (e.g., 2010). If set, word must first appear <= Dec of this year.")
    ap.add_argument("--min_pre_cutoff_months", type=int, default=0,
                    help="Min distinct months a word must appear before cutoff (default 0 = no min)")
    ap.add_argument("--min_pre_cutoff_doc_months", type=int, default=0,
                    help="Min months with docs>0 before cutoff (uses month_docs if present; default 0)")
    args = ap.parse_args()

    cutoff = pd.Timestamp(f"{args.cutoff_year}-01-01")
    strict_deadline = pd.Timestamp(f"{args.strict_year}-12-31") if args.strict_year else None

    panel = load_panel(args.panel)
    raw = json.loads(Path(args.sector_dict_in).read_text())
    sector_dict = {k: [str(w).lower() for w in v] for k, v in raw.items()}
    rev_map = build_reverse_map(sector_dict)
    words = set(rev_map.keys())

    stats = compute_word_stats(panel, words, cutoff)

    # eligibility tests
    eligible = pd.Series(True, index=stats.index)
    # must be seen before cutoff
    eligible &= stats["first_seen_before_cutoff"].notna()
    # stricter year (if provided)
    if strict_deadline is not None:
        eligible &= (stats["first_seen"] <= strict_deadline)

    # min months presence tests
    if "months_before_cutoff" in stats.columns:
        eligible &= (stats["months_before_cutoff"].fillna(0) >= args.min_pre_cutoff_months)
    if "pre_cutoff_doc_months" in stats.columns:
        eligible &= (stats["pre_cutoff_doc_months"].fillna(0) >= args.min_pre_cutoff_doc_months)

    stats["kept"] = eligible.values

    # reason text (simple)
    def reason(row):
        rs = []
        if pd.isna(row["first_seen_before_cutoff"]):
            rs.append("never-seen-before-cutoff")
        if strict_deadline is not None and not pd.isna(row["first_seen"]) and row["first_seen"] > strict_deadline:
            rs.append(f"first-seen-after-strict-{args.strict_year}")
        if row.get("months_before_cutoff", 0) < args.min_pre_cutoff_months:
            rs.append(f"months<{args.min_pre_cutoff_months}")
        if row.get("pre_cutoff_doc_months", 0) < args.min_pre_cutoff_doc_months:
            rs.append(f"doc_months<{args.min_pre_cutoff_doc_months}")
        return ";".join(rs) if rs else "ok"
    stats["reason"] = stats.apply(reason, axis=1)

    # attach sectors for transparency
    stats["sectors"] = stats["word"].map(lambda w: ",".join(sorted(rev_map.get(w, []))))

    # write report
    Path(args.appearance_report_out).parent.mkdir(parents=True, exist_ok=True)
    stats.sort_values(["kept","first_seen"], ascending=[False, True]).to_csv(args.appearance_report_out, index=False)

    # build frozen dict
    kept_words = set(stats.loc[stats["kept"], "word"])
    frozen_rev = {w: rev_map[w] for w in kept_words}
    frozen_fwd = defaultdict(list)
    for w, secs in frozen_rev.items():
        for s in secs:
            frozen_fwd[s].append(w)
    Path(args.sector_dict_out).write_text(json.dumps({k: sorted(set(v)) for k, v in frozen_fwd.items()}, indent=2))

    print("✓ Frozen dictionary:", args.sector_dict_out)
    print("✓ Appearance report:", args.appearance_report_out)
    print(f"Kept {len(kept_words)}/{len(words)} words.")

if __name__ == "__main__":
    main()

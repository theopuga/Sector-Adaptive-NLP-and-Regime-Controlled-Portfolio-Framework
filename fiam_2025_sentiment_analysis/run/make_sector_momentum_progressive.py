#!/usr/bin/env python3
"""
Progressive Sector Momentum Builder
===================================
Purpose:
    Builds **progressive sector word frequency and momentum series** that evolve
    over time with newly activated words, ensuring **out-of-sample integrity**.
    The process integrates both:
      (A) the static, frozen sector dictionary (pre-2015 vocabulary), and  
      (B) newly activated words that enter after their first activation date
          (typically month 13 in mapping results).

Conceptually, this simulates a “real-time expanding lexicon” where new
vocabulary is incorporated only when it becomes observable — producing
a sector-level signal series that remains fully forward-valid.

------------------------------------------------------------
Pipeline Overview:
------------------------------------------------------------
1. **Load Inputs**
   - Word-level panel (`word_trends_monthly.parquet`)
   - Frozen dictionary (JSON from `freeze_sector_dictionary.py`)
   - Activation mapping CSV (from `map_new_words_to_sectors.py` or
     `validate_activation_map.py`)

2. **Filter and Split**
   - Keep only words present in frozen or activation vocabularies.
   - Separate frozen vs new words.
   - For new words, enforce activation start month (no early leakage).

3. **Map & Aggregate**
   - Map grams → sectors using both frozen and activation dictionaries.
   - Split multi-sector words evenly across sectors.
   - Aggregate to sector-level monthly frequency (`sector_tf_share`).

4. **Compute Momentum**
   - Compute rolling sector momentum windows (1M, 3M, 6M, 12M)
     using a numerically stable percentage change function.

5. **Output**
   - `sector_word_frequency_monthly_progressive.parquet`
   - `sector_word_momentum_monthly_progressive.parquet`
   - `sector_momentum_summary_progressive.csv` (top 3 sectors by 6M momentum)

------------------------------------------------------------
Arguments:
------------------------------------------------------------
--panel                  Path to word-trend parquet file.
--frozen_dict            Path to frozen sector dictionary JSON.
--activations            CSV of new word activations with assigned sectors.
--use_share              Column to aggregate ('tf_share' or 'df_share').
--out_dir                Output directory for all progressive files.
--filter_suggest_keep    If set, only include activations flagged as “suggest_keep”.
--min_date               Optional earliest date to include.
--smoke_nrows            Optional limit for quick dry runs (testing).

------------------------------------------------------------
Key Guarantees:
------------------------------------------------------------
- **Leak-free:** New words only contribute after activation month.
- **Fully cumulative:** Frozen dictionary words remain active permanently.
- **Extensible:** Can incorporate multiple activation CSVs or alternative scoring logic.
- **Robust:** Handles missing/ambiguous mappings gracefully.

Example:
python run/make_sector_momentum_progressive.py \
  --panel data/textsec/processed/word_trends_monthly.parquet \
  --frozen_dict data/textsec/processed/FreezeDict/sector_dictionary.json \
  --activations data/textsec/processed/ProgressiveDict/new_words_sector_mapping_active_ALL.csv \
  --use_share tf_share \
  --out_dir data/textsec/processed/ProgressiveDict \
  --filter_suggest_keep

------------------------------------------------------------
Outputs:
------------------------------------------------------------
✓ sector_word_frequency_monthly_progressive.parquet  
✓ sector_word_momentum_monthly_progressive.parquet  
✓ sector_momentum_summary_progressive.csv
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


def log(msg: str) -> None:
    ts = datetime.now().strftime("[%H:%M:%S]")
    print(f"{ts} {msg}")
    sys.stdout.flush()


def to_month_id(dt_series: pd.Series) -> pd.Series:
    s = pd.to_datetime(dt_series)
    return (s.dt.year * 12 + (s.dt.month - 1)).astype("int64")


def load_frozen_reverse_map(frozen_fp: Path) -> dict:
    """sector->words JSON → word->set(sectors), lowercased."""
    obj = json.loads(Path(frozen_fp).read_text())
    rev = {}
    for sector, words in obj.items():
        for w in words:
            lw = str(w).lower()
            rev.setdefault(lw, set()).add(sector)
    return rev


def load_activations(activ_fp: Path, filter_suggest_keep: bool = True) -> tuple[dict, dict]:
    """
    Read activation CSV with columns:
      word, assigned_sector, first_active_date (ISO date),
      [optional flags: ambiguous, stoplisted, suggest_keep]
    Returns:
      word2sectors_new: word->set(sectors)
      word2first_active_mid: word->first_active month_id (int)
    """
    act = pd.read_csv(activ_fp, parse_dates=["first_active_date"])
    # normalize
    act["word"] = act["word"].astype(str).str.lower()

    # optional filtering using validation flag
    if filter_suggest_keep and ("suggest_keep" in act.columns):
        before = len(act)
        act = act[act["suggest_keep"] == True].copy()
        print(f"[INFO] filtered activations: {before:,} → {len(act):,} kept (suggest_keep==True)")
    else:
        print("[INFO] using all activations (no suggest_keep filter applied)")

    # month-id for activation
    act["first_active_mid"] = to_month_id(act["first_active_date"])

    word2sectors_new: dict[str, set] = {}
    word2first_active: dict[str, int] = {}
    for r in act.itertuples(index=False):
        w = r.word
        s = str(r.assigned_sector)
        fa_mid = int(r.first_active_mid)
        word2sectors_new.setdefault(w, set()).add(s)
        prev = word2first_active.get(w)
        word2first_active[w] = fa_mid if prev is None else min(prev, fa_mid)
    return word2sectors_new, word2first_active



def attach_and_split(df: pd.DataFrame, mapper: dict) -> pd.DataFrame:
    """
    Map grams → sector(s), explode, equal-split tf_share across multi-sector words.
    Input df cols: date, gram, tf_share
    Returns cols: date, sector, sector_tf_share
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "sector", "sector_tf_share"])
    out = df.copy()
    out["sectors"] = out["gram"].map(lambda g: sorted(mapper.get(g, set())))
    out = out[out["sectors"].map(len) > 0]
    if out.empty:
        return pd.DataFrame(columns=["date", "sector", "sector_tf_share"])
    out = out.explode("sectors")
    # equal split within (date, gram)
    counts = out.groupby(["date", "gram"])["sectors"].transform("count")
    out["sector_tf_share"] = out["tf_share"] / counts
    return out[["date", "sectors", "sector_tf_share"]].rename(columns={"sectors": "sector"})


def add_momentum(sector_freq: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    if sector_freq.empty:
        for c in ["mom_1m","mom_3m","mom_6m","mom_12m"]:
            sector_freq[c] = np.nan
        return sector_freq

    sector_freq = sector_freq.sort_values(["sector","date"]).copy()
    g = sector_freq.groupby("sector")["sector_tf_share"]

    EPS = 1e-9  # small floor to avoid division by ~0

    def pct_change_safe(x, k):
        prev = x.shift(k)
        # when prev is tiny, treat as missing to avoid blowing up
        denom = prev.where(prev.abs() > EPS, np.nan)
        return (x - prev) / denom

    sector_freq["mom_1m"]  = g.apply(lambda s: pct_change_safe(s, 1)).reset_index(level=0, drop=True)
    sector_freq["mom_3m"]  = g.apply(lambda s: pct_change_safe(s, 3)).reset_index(level=0, drop=True)
    sector_freq["mom_6m"]  = g.apply(lambda s: pct_change_safe(s, 6)).reset_index(level=0, drop=True)
    sector_freq["mom_12m"] = g.apply(lambda s: pct_change_safe(s, 12)).reset_index(level=0, drop=True)

    return sector_freq



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="word_trends_monthly.parquet")
    ap.add_argument("--frozen_dict", required=True, help="sector_dictionary.json (frozen)")
    ap.add_argument("--activations", required=True, help="new_words_sector_mapping_active_ALL.csv")
    ap.add_argument("--use_share", default="tf_share", choices=["tf_share", "df_share"], help="column to aggregate")
    ap.add_argument("--out_dir", default="data/textsec/processed", help="output directory")
    ap.add_argument("--min_date", default=None, help="optional YYYY-MM-DD to drop earlier rows")
    ap.add_argument("--smoke_nrows", type=int, default=None, help="optional limit for quick smoke test")
    ap.add_argument("--filter_suggest_keep", action="store_true",
                help="If set, only use activations where suggest_keep == True (from validate_activation_map.py).")

    args = ap.parse_args()

    panel_fp = Path(args.panel)
    frozen_fp = Path(args.frozen_dict)
    activ_fp = Path(args.activations)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("loading panel…")
    use_cols = ["date", "gram", args.use_share]
    panel = pd.read_parquet(panel_fp, columns=use_cols)
    panel = panel.rename(columns={args.use_share: "tf_share"})
    if args.smoke_nrows:
        panel = panel.iloc[: args.smoke_nrows].copy()
    if args.min_date:
        panel = panel[pd.to_datetime(panel["date"]) >= pd.to_datetime(args.min_date)].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["gram"] = panel["gram"].astype(str).str.lower()

    log(f"panel rows: {len(panel):,} | unique grams: {panel['gram'].nunique():,}")

    log("loading frozen dictionary…")
    word2secs_frozen = load_frozen_reverse_map(frozen_fp)
    frozen_vocab = set(word2secs_frozen.keys())
    log(f"frozen vocab size: {len(frozen_vocab):,}")

    log("loading activations (ALL words, no filtering)…")
    word2secs_new, word2first_active = load_activations(activ_fp, filter_suggest_keep=args.filter_suggest_keep)

    new_vocab = set(word2secs_new.keys())
    log(f"activation vocab size: {len(new_vocab):,}")

    # Cut panel down to only words we actually need (big speed/memory win)
    want_vocab = frozen_vocab | new_vocab
    panel = panel[panel["gram"].isin(want_vocab)].copy()
    log(f"panel after vocab filter: {len(panel):,} rows")

    # Split into frozen vs new; gate new by activation month_id
    panel["month_id"] = to_month_id(panel["date"])
    is_frozen = panel["gram"].isin(frozen_vocab)
    is_new = panel["gram"].isin(new_vocab)

    panel_frozen = panel[is_frozen][["date", "gram", "tf_share"]].copy()
    panel_new = panel[is_new][["date", "gram", "tf_share", "month_id"]].copy()

    # Gate new words to month_id >= first_active_mid
    panel_new["first_active_mid"] = panel_new["gram"].map(word2first_active).astype("Int64")
    panel_new = panel_new[panel_new["month_id"] >= panel_new["first_active_mid"]].copy()
    panel_new = panel_new.drop(columns=["month_id", "first_active_mid"])

    log(f"frozen rows: {len(panel_frozen):,} | new (activated) rows: {len(panel_new):,}")

    # Map to sectors and split
    log("mapping + splitting frozen words…")
    sf_frozen = attach_and_split(panel_frozen, word2secs_frozen)
    log(f"mapped frozen rows: {len(sf_frozen):,}")

    log("mapping + splitting new words…")
    sf_new = attach_and_split(panel_new, word2secs_new)
    log(f"mapped new rows: {len(sf_new):,}")

    # Combine and aggregate to (date, sector)
    log("aggregating sector frequency…")
    sector_freq = (
        pd.concat([sf_frozen, sf_new], ignore_index=True)
        .groupby(["date", "sector"], as_index=False)["sector_tf_share"].sum()
        .sort_values(["sector", "date"])
    )

    # Momentum
    log("computing momentum windows…")
    sector_mom = add_momentum(sector_freq)

    # Save
    freq_fp = out_dir / "sector_word_frequency_monthly_progressive.parquet"
    mom_fp = out_dir / "sector_word_momentum_monthly_progressive.parquet"
    sector_mom.to_parquet(freq_fp, index=False)
    sector_mom.to_parquet(mom_fp, index=False)

    # Summary top-3 by 6m momentum each month
    log("building summary (top-3 by 6m)…")
    summary = (
        sector_mom.dropna(subset=["mom_6m"])
        .sort_values(["date", "mom_6m"], ascending=[True, False])
        .groupby("date")
        .head(3)[["date", "sector", "mom_6m"]]
    )
    summary_fp = out_dir / "sector_momentum_summary_progressive.csv"
    summary.to_csv(summary_fp, index=False)

    log("done.")
    print("Wrote:")
    print(" -", freq_fp)
    print(" -", mom_fp)
    print(" -", summary_fp)


if __name__ == "__main__":
    main()

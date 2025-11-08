#!/usr/bin/env python3
"""
validate_activation_map.py

Purpose
-------
Validate and annotate the raw new-word→sector mapping produced by
`map_new_words_to_sectors.py` without filtering any rows.

What it does (non-destructive):
- Ensures each word has:
    * first_seen_date  (derived from month_id if needed)
    * first_active_date = first_seen_date + 12 months (no-lookahead activation)
- Adds quality flags:
    * pearson_ok       (pearson >= P_MIN)
    * margin_ok        (margin_pearson >= M_MIN)
    * stoplisted       (matches static generic patterns/tokens if provided)
    * suggest_keep     (!ambiguous & pearson_ok & margin_ok & !stoplisted)
- Optional: enforce a minimum activation year (e.g., 2015) safely.

Outputs
-------
1) --out (required): activation map used downstream
   Columns: word, assigned_sector, first_active_date, ambiguous, pearson,
            margin_pearson, stoplisted, suggest_keep
2) --reco_out (optional): full, human-readable table with extra context
   Columns: + first_seen, first_seen_date, plus all above

Example
-------
python run/validate_activation_map.py \
  --inp  data/textsec/processed/ProgressiveDict/new_words_sector_mapping_fixed12m.csv \
  --out  data/textsec/processed/ProgressiveDict/new_words_sector_mapping_active_ALL.csv \
  --reco_out data/textsec/processed/ProgressiveDict/new_words_sector_mapping_recommendations.csv \
  --p_min 0.30 --m_min 0.05 --cutoff_year 2015 \
  --enforce_cutoff
"""

import argparse
from datetime import datetime
from pathlib import Path
import json
import re

import pandas as pd
from dateutil.relativedelta import relativedelta


def month_id_to_date(mid: int) -> datetime:
    """Convert YYYY*12 + (MM-1) month_id -> datetime(YYYY, MM, 1)."""
    y = int(mid) // 12
    m = (int(mid) % 12) + 1
    return datetime(y, m, 1)


def load_stop_cfg(path: str | None):
    """Load optional generic stopword config with 'patterns' and 'tokens'."""
    if not path:
        return None, set()
    p = Path(path)
    if not p.exists():
        return None, set()
    cfg = json.loads(p.read_text())
    pats = cfg.get("patterns", [])
    toks = {str(t).lower() for t in cfg.get("tokens", [])}
    regex = re.compile("|".join(pats), re.IGNORECASE) if pats else None
    return regex, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="new_words_sector_mapping_fixed12m.csv")
    ap.add_argument("--out", required=True, help="activation file to write (ALL words)")
    ap.add_argument("--reco_out", default=None, help="optional recommendations CSV (full annotated table)")
    ap.add_argument("--p_min", type=float, default=0.30, help="suggested min pearson for suggest_keep flag")
    ap.add_argument("--m_min", type=float, default=0.05, help="suggested min margin for suggest_keep flag")
    ap.add_argument("--cutoff_year", type=int, default=2015, help="enforced minimum activation year if --enforce_cutoff")
    ap.add_argument("--stopwords_json", default=None, help="optional JSON with patterns/tokens to flag as stoplisted")
    ap.add_argument("--enforce_cutoff", action="store_true", help="if set, clamp first_active_date to >= Jan 1 cutoff_year")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.reco_out:
        Path(args.reco_out).parent.mkdir(parents=True, exist_ok=True)

    # 1) Load mapping
    m = pd.read_csv(inp)
    # Normalize column names we expect
    cols = set(c.lower() for c in m.columns)
    # Required minimal set for raw mapping
    required = {"word", "assigned_sector", "pearson", "margin_pearson", "ambiguous"}
    missing = required - cols
    if missing:
        raise ValueError(f"Input file missing required columns: {missing}")

    # 2) Ensure lowercase word for consistent joins later (but keep original column)
    if "word_lc" not in m.columns:
        m["word_lc"] = m["word"].astype(str).str.lower()

    # 3) First-seen date derivation
    # Accept either 'first_seen' (month_id int) or 'first_seen_date' (ISO date)
    if "first_seen_date" in m.columns:
        m["first_seen_date"] = pd.to_datetime(m["first_seen_date"])
    elif "first_seen" in m.columns:
        m["first_seen_date"] = m["first_seen"].apply(month_id_to_date)
    else:
        raise ValueError("Input must contain either 'first_seen' (month_id) or 'first_seen_date'.")

    # 4) Compute first_active_date = first_seen_date + 12 months (no-lookahead)
    m["first_active_date"] = m["first_seen_date"].apply(lambda d: d + relativedelta(months=12))

    # 5) Optional cutoff enforcement (still non-destructive to mapping; only shifts activation forward)
    if args.enforce_cutoff:
        cutoff_dt = datetime(args.cutoff_year, 1, 1)
        m["first_active_date"] = m["first_active_date"].apply(lambda d: max(d, cutoff_dt))

    # 6) Load optional static stopwords (for flagging only, not filtering)
    gen_re, toks = load_stop_cfg(args.stopwords_json)
    if gen_re is not None or toks:
        g = m["word_lc"]
        stop_mask = pd.Series(False, index=m.index)
        if gen_re is not None:
            stop_mask |= g.str.contains(gen_re)
        if toks:
            stop_mask |= g.isin(toks)
        m["stoplisted"] = stop_mask
    else:
        m["stoplisted"] = False

    # 7) Quality flags (non-binding)
    m["pearson_ok"] = m["pearson"] >= args.p_min
    m["margin_ok"] = m["margin_pearson"].fillna(0) >= args.m_min
    # Ensure ambiguous exists as boolean
    if m["ambiguous"].dtype != bool:
        m["ambiguous"] = m["ambiguous"].astype(bool)
    m["suggest_keep"] = (~m["ambiguous"]) & m["pearson_ok"] & m["margin_ok"] & (~m["stoplisted"])

    # 8) Write activation file (ALL words; this is the one the progressive builder uses)
    act_cols = [
        "word", "assigned_sector", "first_active_date",
        "ambiguous", "pearson", "margin_pearson",
        "stoplisted", "suggest_keep"
    ]
    m[act_cols].to_csv(out, index=False)

    # 9) Optional recommendations table (more context for humans)
    if args.reco_out:
        reco_cols = [
            "word", "assigned_sector",
            "first_seen", "first_seen_date", "first_active_date",
            "pearson", "margin_pearson", "ambiguous",
            "stoplisted", "pearson_ok", "margin_ok", "suggest_keep"
        ]
        # Keep columns that exist
        reco_cols = [c for c in reco_cols if c in m.columns]
        m[reco_cols].to_csv(args.reco_out, index=False)

    # 10) Console summary
    total = len(m)
    n_amb = int(m["ambiguous"].sum())
    n_stop = int(m["stoplisted"].sum())
    n_sug = int(m["suggest_keep"].sum())
    print(f"Rows: {total:,} | ambiguous: {n_amb:,} ({n_amb/total:.1%}) | stoplisted: {n_stop:,} ({n_stop/total:.1%}) | suggest_keep: {n_sug:,} ({n_sug/total:.1%})")
    print(f"✓ wrote activation map: {out}")
    if args.reco_out:
        print(f"✓ wrote recommendations: {args.reco_out}")


if __name__ == "__main__":
    main()

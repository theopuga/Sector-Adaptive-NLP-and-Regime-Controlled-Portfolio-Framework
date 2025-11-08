#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, sys, os, re, json, yaml, multiprocessing, traceback, time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
import numpy as np

# ---------------- CUDA-safe multiprocessing ----------------
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

MODEL_NAME = "yiyanghkust/finbert-tone"

# ---------------- logging ----------------
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def eprint(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

# ---------------- helpers ----------------
def read_paths(paths_yaml: str) -> Dict:
    if not paths_yaml:
        return {}
    with open(paths_yaml, "r") as f:
        return yaml.safe_load(f) or {}

def _discover_shards(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())

def _load_manifest(path: Path) -> pl.DataFrame:
    import pandas as pd
    man = pd.read_csv(path)
    col = "path" if "path" in man.columns else ("shard_path" if "shard_path" in man.columns else None)
    if not col:
        raise ValueError("Manifest must have 'path' or 'shard_path'")
    return pl.from_pandas(man).rename({col: "__shard_path"})

def _apply_exclude_regex(man: pl.DataFrame, pattern: str) -> pl.DataFrame:
    if not pattern:
        return man
    return man.filter(~pl.col("__shard_path").str.contains(pattern))

def _filter_scope(man: pl.DataFrame, scope: str, scope_value: str) -> pl.DataFrame:
    if scope == "global":
        return man
    path_col = pl.col("__shard_path").str.replace("\\\\", "/")
    if scope == "global_ex_us":
        return man.filter(~path_col.str.contains(r"/USA/"))
    if scope == "country" and scope_value:
        pat = fr"/{re.escape(scope_value)}/"
        return man.filter(path_col.str.contains(pat))
    if scope == "continent" and scope_value:
        pat = fr"/{re.escape(scope_value)}/"
        return man.filter(path_col.str.contains(pat))
    return man

def _normalize_id_columns(df: pl.DataFrame) -> pl.DataFrame:
    if "gvkey" in df.columns:
        df = df.with_columns(pl.col("gvkey").cast(pl.Utf8))
    if "iid" in df.columns:
        df = df.with_columns(pl.col("iid").cast(pl.Utf8))
    return df

def _normalize_month_end(df: pl.DataFrame) -> pl.DataFrame:
    if "month_end" not in df.columns:
        return df

    s = df["month_end"]
    # If already a Date -> keep
    if s.dtype == pl.Date:
        return df

    # Datetime -> downcast to Date
    if s.dtype == pl.Datetime:
        return df.with_columns(pl.col("month_end").dt.date())

    # Ints like 202403 or 20240331
    if s.dtype.is_integer():
        return df.with_columns(
            pl.when(pl.col("month_end") >= 10_000_000)  # YYYYMMDD
              .then(
                  pl.col("month_end")
                    .cast(pl.Utf8)
                    .str.strptime(pl.Date, fmt="%Y%m%d", strict=False)
              )
              .otherwise(
                  # assume YYYYMM -> build last day of month
                  pl.date(
                      (pl.col("month_end") // 100).cast(pl.Int32),          # year
                      (pl.col("month_end") % 100).cast(pl.Int32),           # month
                      pl.lit(1, dtype=pl.Int32)
                  ).dt.month_end()  # requires polars >= 0.20; if older, swap for .dt.truncate("1mo").dt.offset_by("1mo") - pl.duration(days=1)
              )
              .alias("month_end")
        )

    # Strings: try common formats explicitly
    if s.dtype == pl.Utf8:
        df_try = df.with_columns(pl.col("month_end").str.strptime(pl.Date, fmt="%Y-%m-%d", strict=True))
        if df_try["month_end"].null_count() == 0:
            return df_try

        df_try = df.with_columns(pl.col("month_end").str.strptime(pl.Date, fmt="%Y%m%d", strict=True))
        if df_try["month_end"].null_count() == 0:
            return df_try

        # "YYYY-MM" -> make it last day of month
        df_try = df.with_columns(
            pl.col("month_end").str.strptime(pl.Date, fmt="%Y-%m", strict=True).dt.month_end()
        )
        if df_try["month_end"].null_count() == 0:
            return df_try

        # "YYYYMM" -> last day of month
        df_try = df.with_columns(
            pl.col("month_end").str.strptime(pl.Date, fmt="%Y%m", strict=True)
              .dt.month_end()
        )
        if df_try["month_end"].null_count() == 0:
            return df_try

        # last resort: non-strict
        return df.with_columns(pl.col("month_end").str.strptime(pl.Date, strict=False))

    # Unknown type: leave as is
    return df
def _ensure_month_end(df: pl.DataFrame) -> pl.DataFrame:
    # If month_end missing, derive from filing_date (assumes YYYY-MM-DD)
    if "month_end" not in df.columns and "filing_date" in df.columns:
        try:
            dt = pl.col("filing_date").str.strptime(pl.Date, strict=False)
            year = dt.dt.year()
            month = dt.dt.month()
            # month end: go to first of next month then subtract one day
            first_next = (year + (month == 12).cast(pl.Int32) * 1).alias("y_tmp"), ((month % 12) + 1).alias("m_tmp")
            # Build a date safely
            df = df.with_columns(
                dt.alias("_dt"),
            )
            # Use Polars month_end function if available; fallback manual
            if hasattr(pl.Expr.dt, "month_end"):
                df = df.with_columns(pl.col("_dt").dt.month_end().alias("month_end")).drop("_dt")
            else:
                # simple approx: last day of month via month start of next minus 1 day
                df = df.with_columns(
                    pl.col("_dt").dt.replace(day=1).dt.offset_by("1mo").dt.offset_by("-1d").alias("month_end")
                ).drop("_dt")
        except Exception:
            df = df.with_columns(pl.lit(None).alias("month_end"))
    return df

def diagnostics(df: pl.DataFrame, lag_days: int) -> Dict:
    return {
        "rows": len(df),
        "lag_days": lag_days,
        "has_text": "text" in df.columns,
    }

def filter_for_scoring(df: pl.DataFrame, lag_days: int) -> pl.DataFrame:
    if "text" not in df.columns:
        return pl.DataFrame()
    return df.filter(pl.col("text").is_not_null() & (pl.col("text").str.len_chars() > 0))

def score_doc_text(
    text: str,
    tokenizer,
    model,
    device,
    max_length: int = 128,
):
    import torch, scipy.special
    with torch.no_grad():
        enc = tokenizer([text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().to("cpu").numpy()
    probs = scipy.special.softmax(logits, axis=1)[0]  # 0=neg,1=neu,2=pos
    return {
        "prob_neg_mean": float(probs[0]),
        "prob_neu_mean": float(probs[1]),
        "prob_pos_mean": float(probs[2]),
        "sent_pos_minus_neg_mean": float(probs[2] - probs[0]),
        "doc_sent_count": 1,
    }

def aggregate_monthly(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate per (gvkey, iid, month_end, country, continent).
    Assumes df has those columns plus the FinBERT-derived metrics.
    """
    df = _normalize_ids_and_types(_normalize_month_end(df))

    # If textsec rows are per-document, you might pre-aggregate here to monthly.
    # Below computes monthly means and sums over the doc-level rows.
    agg_exprs = [
        pl.len().alias("doc_sent_count"),
        pl.mean("prob_neg_mean").alias("prob_neg_mean"),
        pl.mean("prob_neu_mean").alias("prob_neu_mean"),
        pl.mean("prob_pos_mean").alias("prob_pos_mean"),
        pl.mean("sent_pos_minus_neg_mean").alias("sent_pos_minus_neg_mean"),
        pl.first("company_name").alias("company_name"),
        pl.first("excntry").alias("excntry"),
    ]
    
    # Only include ratio columns if they exist
    if "ratio_sent_pos_gt_neg" in df.columns:
        agg_exprs.append(pl.mean("ratio_sent_pos_gt_neg").alias("ratio_sent_pos_gt_neg"))
    if "ratio_sent_neg_gt_pos" in df.columns:
        agg_exprs.append(pl.mean("ratio_sent_neg_gt_pos").alias("ratio_sent_neg_gt_pos"))
    
    agg = (
        df.group_by("gvkey", "iid", "month_end", "country", "continent")
          .agg(agg_exprs)
          .sort("month_end", "country", "gvkey", "iid")
    )
    return agg

def _compute_sec_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add requested SEC sentiment features on monthly aggregated data.
    Expects columns: gvkey, iid, month_end, sent_pos_minus_neg_mean,
    prob_neg_mean, prob_neu_mean, prob_pos_mean, doc_sent_count, etc.
    Computes time-series features per (gvkey, iid, section) when section is present,
    otherwise per (gvkey, iid). Cross-sectional features computed per (month_end, section)
    when section is present, otherwise per month_end.
    """
    if df.is_empty():
        return df

    # Check if we have section column
    has_section = "section" in df.columns and df["section"].null_count() < len(df)
    
    # Ensure proper types/order
    sort_cols = ["gvkey","iid","month_end"]
    if has_section:
        sort_cols.insert(2, "section")  # Insert section after iid, before month_end
    
    df = df.with_columns([
        pl.col("month_end").cast(pl.Date),
        pl.col("gvkey").cast(pl.Utf8),
        pl.col("iid").cast(pl.Utf8),
    ]).sort(sort_cols)  # ordering matters for rolling

    base = pl.col("sent_pos_minus_neg_mean")
    # Group by entity and section (if present) for time-series features
    by_entity = ["gvkey","iid"]
    if has_section:
        by_entity.append("section")

    # Helper windowed columns over entity (and section if present), respecting sort order
    def win(expr: pl.Expr) -> pl.Expr:
        return expr.over(by_entity)

    # Time-series features on base sentiment
    S_mom_1 = (base - base.shift(1)).alias("S_mom_1")
    S_mom_3 = (base - base.shift(3)).alias("S_mom_3")
    S_mom_6 = (base - base.shift(6)).alias("S_mom_6")

    S_pctchg_1 = ((base / base.shift(1)) - 1.0).alias("S_pctchg_1")
    S_sign = pl.when((base - base.shift(1)) > 0).then(1).when((base - base.shift(1)) < 0).then(-1).otherwise(0).alias("S_sign")

    S_ma_3 = base.rolling_mean(window_size=3, min_periods=1).alias("S_ma_3")
    S_ma_6 = base.rolling_mean(window_size=6, min_periods=1).alias("S_ma_6")
    S_ma_12 = base.rolling_mean(window_size=12, min_periods=1).alias("S_ma_12")

    S_vol_3 = base.rolling_std(window_size=3, min_periods=2).alias("S_vol_3")
    S_vol_6 = base.rolling_std(window_size=6, min_periods=2).alias("S_vol_6")
    S_vol_12 = base.rolling_std(window_size=12, min_periods=2).alias("S_vol_12")

    # EMA and MACD family
    S_ema_6 = base.ewm_mean(span=6, adjust=False).alias("S_ema_6")
    S_ema_12 = base.ewm_mean(span=12, adjust=False).alias("S_ema_12")
    macd = (base.ewm_mean(span=12, adjust=False) - base.ewm_mean(span=26, adjust=False)).alias("S_macd_12_26")
    macd_sig = pl.col("S_macd_12_26").ewm_mean(span=9, adjust=False).alias("S_macd_sig_9")

    # Rolling z and CV
    mean_12 = base.rolling_mean(window_size=12, min_periods=3)
    std_12 = base.rolling_std(window_size=12, min_periods=3)
    S_z_12 = ((base - mean_12) / std_12).alias("S_z_12")
    mean_6 = base.rolling_mean(window_size=6, min_periods=3)
    std_6 = base.rolling_std(window_size=6, min_periods=3)
    S_cv_6 = (std_6 / mean_6).alias("S_cv_6")

    # Probability-derived features
    probs = ["prob_neg_mean","prob_neu_mean","prob_pos_mean"]
    # Entropy: -sum p*log(p); clamp probs to avoid nan
    eps = 1e-9
    P_entropy = (
        -(
            (pl.col("prob_neg_mean") + eps).log() * pl.col("prob_neg_mean") +
            (pl.col("prob_neu_mean") + eps).log() * pl.col("prob_neu_mean") +
            (pl.col("prob_pos_mean") + eps).log() * pl.col("prob_pos_mean")
        )
    ).alias("P_entropy")
    prob_confidence = pl.max_horizontal(*[pl.col(c) for c in probs]).alias("prob_confidence")

    # Doc count lag and intensity
    doc_sent_count_lag1 = pl.col("doc_sent_count").shift(1).alias("doc_sent_count_lag1")
    doc_sent_intensity_3 = pl.col("doc_sent_count").rolling_sum(window_size=3, min_periods=1).alias("doc_sent_intensity_3")

    # Apply per-entity window calcs
    df = (
        df.sort(sort_cols)  # ensure order within groups (includes section if present)
          .with_columns([
              win(S_mom_1), win(S_mom_3), win(S_mom_6),
              win(S_pctchg_1), win(S_sign),
              win(S_ma_3), win(S_ma_6), win(S_ma_12),
              win(S_vol_3), win(S_vol_6), win(S_vol_12),
              win(S_ema_6), win(S_ema_12),
          ])
          .with_columns([win(macd)])
    )
    # macd signal depends on macd column now present
    df = df.with_columns([win(macd_sig)])

    # Remaining per-entity windows
    df = df.with_columns([
        win(S_z_12),
        win(S_cv_6),
        P_entropy,
        prob_confidence,
        win(doc_sent_count_lag1),
        win(doc_sent_intensity_3),
    ])

    # Cross-sectional features per month_end (and section if present) on base sentiment
    # Group for cross-sectional features
    xs_group = ["month_end"]
    if has_section:
        xs_group.append("section")
    
    # Rank within month (and section if present) (ascending rank: lower sentiment -> lower rank)
    df = df.with_columns([
        pl.col("sent_pos_minus_neg_mean").rank("average").over(xs_group).alias("S_xs_rank"),
    ])

    # Cross-sectional z within month (and section if present)
    cs_mean = pl.col("sent_pos_minus_neg_mean").mean().over(xs_group)
    cs_std = pl.col("sent_pos_minus_neg_mean").std().over(xs_group)
    df = df.with_columns([
        ((pl.col("sent_pos_minus_neg_mean") - cs_mean) / cs_std).alias("S_xs_z"),
    ])

    return df

# ---------------- worker ----------------
def process_shard(
    shard_path: str,
    device_str: str,
    batch_size: int,
    max_length: int,
    lag_days: int,
    checkpoint_dir: str,
    fp16: bool,
    is_usa_scope: bool = False,
) -> Optional[Dict]:
    try:
        # Lazy imports in the worker
        import traceback
        from pathlib import Path
        import polars as pl
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # One model per process
        device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        if fp16 and device.type == "cuda":
            model = model.half()
        model = model.to(device).eval()

        p = Path(shard_path)
        if not p.exists():
            return {"shard": shard_path, "status": "missing", "error": "File not found"}

        df = pl.read_parquet(p)

        # --- Handle USA shards with separate rf/mgmt columns ---
        # Check if we have rf, mgmt, or text columns
        has_rf = "rf" in df.columns
        has_mgmt = "mgmt" in df.columns
        has_text = "text" in df.columns
        
        if (has_rf or has_mgmt) and not has_text:
            # USA format: convert rf/mgmt into unified (section, text) format
            parts = []
            
            # Common columns to preserve
            common_cols = [c for c in df.columns if c not in ("rf", "mgmt", "text", "section")]
            
            if has_rf:
                df_rf = df.select([
                    *[pl.col(c) for c in common_cols],
                    pl.col("rf").cast(pl.Utf8).alias("text"),
                    pl.lit("risk_factors").alias("section"),
                ]).filter(pl.col("text").is_not_null() & (pl.col("text").str.len_chars() > 0))
                parts.append(df_rf)
            
            if has_mgmt:
                df_mgmt = df.select([
                    *[pl.col(c) for c in common_cols],
                    pl.col("mgmt").cast(pl.Utf8).alias("text"),
                    pl.lit("management_discussion_and_analysis").alias("section"),
                ]).filter(pl.col("text").is_not_null() & (pl.col("text").str.len_chars() > 0))
                parts.append(df_mgmt)
            
            # If both rf and mgmt exist, optionally create a fused version
            # (skip for now - can add later if needed)
            
            if parts:
                df = pl.concat(parts, how="vertical_relaxed")
        
        elif has_text:
            # Standard format: ensure section column exists if not present
            if "section" not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("section"))
            # If section is null/empty but text exists, mark as full document
            if df["section"].null_count() == len(df) or df.filter(pl.col("section").is_not_null()).height == 0:
                df = df.with_columns(
                    pl.when(pl.col("text").is_not_null() & (pl.col("text").str.len_chars() > 0))
                       .then(pl.lit("fused_full_document"))
                       .otherwise(pl.lit(None).cast(pl.Utf8))
                       .alias("section")
                )
        
        # --- Ensure required columns exist (fill with None if missing) ---
        need = ["gvkey","iid","company_name","filing_date","text","section","country","continent","excntry"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            df = df.with_columns([pl.lit(None).alias(c) for c in missing])

        # --- Normalize IDs/types + month_end EARLY ---
        # gvkey: try Int cast (handles float/int/num-strings), fall back to original, then to Utf8
        if "gvkey" in df.columns:
            df = df.with_columns(
                pl.coalesce([
                    pl.col("gvkey").cast(pl.Int64, strict=False),
                    pl.col("gvkey"),
                ])
                .cast(pl.Utf8)
                .alias("gvkey")
            )
        if "iid" in df.columns:
            df = df.with_columns(pl.col("iid").cast(pl.Utf8).str.strip_chars())

        for txt in ("company_name","country","continent","excntry","section","text"):
            if txt in df.columns:
                df = df.with_columns(pl.col(txt).cast(pl.Utf8))

        # Month-end normalization (unchanged logic; robust parsing & rebuild)
        def _rebuild_me(_df: pl.DataFrame) -> pl.DataFrame:
            # filing_date preferred
            if "filing_date" in _df.columns:
                s = _df["filing_date"]
                if s.dtype == pl.Datetime:
                    return _df.with_columns(pl.col("filing_date").dt.date().dt.month_end().alias("month_end"))
                if s.dtype == pl.Date:
                    return _df.with_columns(pl.col("filing_date").dt.month_end().alias("month_end"))
                if s.dtype == pl.Utf8:
                    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
                        t = _df.with_columns(pl.col("filing_date").str.strptime(pl.Date, fmt=fmt, strict=True))
                        if t["filing_date"].null_count() < len(t):
                            return t.with_columns(pl.col("filing_date").dt.month_end().alias("month_end"))
                    t = _df.with_columns(pl.col("filing_date").str.strptime(pl.Date, strict=False))
                    if t["filing_date"].null_count() < len(t):
                        return t.with_columns(pl.col("filing_date").dt.month_end().alias("month_end"))
            # year + month
            if "year" in _df.columns and "month" in _df.columns:
                t = _df.with_columns([
                    pl.col("year").cast(pl.Int32, strict=False).alias("year__i"),
                    pl.col("month").cast(pl.Int32, strict=False).alias("month__i"),
                ])
                if t["year__i"].null_count() < len(t) and t["month__i"].null_count() < len(t):
                    return t.with_columns(
                        pl.date(pl.col("year__i"), pl.col("month__i"), pl.lit(1, dtype=pl.Int32)).dt.month_end().alias("month_end")
                    )
            # other common date-like columns
            for cand in ("report_date","date","doc_date","period_end"):
                if cand in _df.columns:
                    s = _df[cand]
                    if s.dtype == pl.Datetime:
                        return _df.with_columns(pl.col(cand).dt.date().dt.month_end().alias("month_end"))
                    if s.dtype == pl.Date:
                        return _df.with_columns(pl.col(cand).dt.month_end().alias("month_end"))
                    if s.dtype == pl.Utf8:
                        for fmt in ("%Y-%m-%d","%Y%m%d","%Y/%m/%d"):
                            t = _df.with_columns(pl.col(cand).str.strptime(pl.Date, fmt=fmt, strict=True))
                            if t[cand].null_count() < len(t):
                                return t.with_columns(pl.col(cand).dt.month_end().alias("month_end"))
                        t = _df.with_columns(pl.col(cand).str.strptime(pl.Date, strict=False))
                        if t[cand].null_count() < len(t):
                            return t.with_columns(pl.col(cand).dt.month_end().alias("month_end"))
            # fallback: create null Date column
            return _df.with_columns(pl.lit(None, dtype=pl.Date).alias("month_end"))

        if "month_end" in df.columns:
            s = df["month_end"]
            if s.dtype == pl.Datetime:
                df = df.with_columns(pl.col("month_end").dt.date())
            elif s.dtype == pl.Utf8:
                ok = False
                for fmt in ("%Y-%m-%d","%Y%m%d"):
                    t = df.with_columns(pl.col("month_end").str.strptime(pl.Date, fmt=fmt, strict=True))
                    if t["month_end"].null_count() == 0:
                        df, ok = t, True
                        break
                if not ok:
                    t = df.with_columns(pl.col("month_end").str.strptime(pl.Date, fmt="%Y-%m", strict=True).dt.month_end())
                    if t["month_end"].null_count() == 0:
                        df, ok = t, True
                if not ok:
                    t = df.with_columns(pl.col("month_end").str.strptime(pl.Date, fmt="%Y%m", strict=True).dt.month_end())
                    if t["month_end"].null_count() == 0:
                        df, ok = t, True
                if not ok:
                    df = df.with_columns(pl.col("month_end").str.strptime(pl.Date, strict=False))
            if df["month_end"].null_count() == len(df):
                df = _rebuild_me(df)
        else:
            df = _rebuild_me(df)

        # Filter docs to score
        d = filter_for_scoring(df, lag_days)
        if d.is_empty():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return {"shard": shard_path, "status": "empty_after_filter", "diagnostics": diagnostics(df, lag_days)}

        # --------- SCORING ----------
        rows = []
        texts = d["text"].to_list()
        d_dicts = d.to_dicts()  # Convert once for efficiency
        for i in range(0, len(texts), max(1, batch_size)):
            batch = texts[i:i+batch_size]
            for j, t in enumerate(batch):
                row_idx = i + j
                if row_idx >= len(d_dicts):
                    continue
                met = score_doc_text(t, tokenizer, model, device, max_length=max_length)
                if met is None:
                    continue
                row = d_dicts[row_idx]
                rows.append({
                    "gvkey": row.get("gvkey"),
                    "iid": row.get("iid",""),
                    "company_name": row.get("company_name",""),
                    "excntry": row.get("excntry",""),
                    "country": row.get("country",""),
                    "continent": row.get("continent",""),
                    "section": row.get("section",""),
                    "filing_date": row.get("filing_date"),
                    "month_end": row.get("month_end"),
                    **met
                })

        if not rows:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return {"shard": shard_path, "status": "no_valid_docs", "diagnostics": diagnostics(df, lag_days)}

        # --------- ROBUST DF CONSTRUCTION ----------
        def _safe_df_from_rows(rows: list[dict]) -> pl.DataFrame:
            if not rows:
                return pl.DataFrame([])
            df2 = pl.DataFrame(rows, infer_schema_length=max(1000, len(rows)))

            if "gvkey" in df2.columns:
                df2 = df2.with_columns(
                    pl.coalesce([
                        pl.col("gvkey").cast(pl.Int64, strict=False),
                        pl.col("gvkey"),
                    ])
                    .cast(pl.Utf8)
                    .alias("gvkey")
                )
            if "iid" in df2.columns:
                df2 = df2.with_columns(pl.col("iid").cast(pl.Utf8).str.strip_chars())

            if "doc_sent_count" in df2.columns:
                df2 = df2.with_columns(pl.col("doc_sent_count").cast(pl.Int64))

            float_cols = [
                "prob_neg_mean","prob_neu_mean","prob_pos_mean",
                "sent_pos_minus_neg_mean",
            ]
            # Only add ratio columns if they exist
            for ratio_col in ["ratio_sent_pos_gt_neg", "ratio_sent_neg_gt_pos"]:
                if ratio_col in df2.columns:
                    float_cols.append(ratio_col)
            present = [c for c in float_cols if c in df2.columns]
            if present:
                df2 = df2.with_columns([pl.col(c).cast(pl.Float64) for c in present])

            for txt in ("company_name","country","continent","excntry"):
                if txt in df2.columns:
                    df2 = df2.with_columns(pl.col(txt).cast(pl.Utf8))

            if "month_end" not in df2.columns or df2["month_end"].null_count() == len(df2):
                if "filing_date" in df2.columns:
                    s = df2["filing_date"]
                    if s.dtype == pl.Datetime:
                        df2 = df2.with_columns(pl.col("filing_date").dt.date().dt.month_end().alias("month_end"))
                    elif s.dtype == pl.Date:
                        df2 = df2.with_columns(pl.col("filing_date").dt.month_end().alias("month_end"))
            return df2


        df_rows = _safe_df_from_rows(rows)

        # Build aggregation expressions
        agg_exprs = [
            pl.len().alias("doc_sent_count"),
            pl.mean("prob_neg_mean").alias("prob_neg_mean"),
            pl.mean("prob_neu_mean").alias("prob_neu_mean"),
            pl.mean("prob_pos_mean").alias("prob_pos_mean"),
            pl.mean("sent_pos_minus_neg_mean").alias("sent_pos_minus_neg_mean"),
            pl.first("company_name").alias("company_name"),
            pl.first("excntry").alias("excntry"),
        ]
        # Only include ratio columns if they exist
        if "ratio_sent_pos_gt_neg" in df_rows.columns:
            agg_exprs.append(pl.mean("ratio_sent_pos_gt_neg").alias("ratio_sent_pos_gt_neg"))
        if "ratio_sent_neg_gt_pos" in df_rows.columns:
            agg_exprs.append(pl.mean("ratio_sent_neg_gt_pos").alias("ratio_sent_neg_gt_pos"))
        
        # USA scope: aggregate by section separately, then create totals
        if is_usa_scope and "section" in df_rows.columns and df_rows["section"].null_count() < len(df_rows):
            # Check if we have valid section data
            has_sections = df_rows.filter(pl.col("section").is_not_null() & (pl.col("section").str.len_chars() > 0)).height > 0
            
            if has_sections:
                # Per-section aggregations
                monthly_by_section = (
                    df_rows.group_by("gvkey","iid","month_end","country","continent","section")
                           .agg(agg_exprs)
                )
                
                # Total aggregations (across all sections)
                monthly_total = (
                    df_rows.group_by("gvkey","iid","month_end","country","continent")
                           .agg(agg_exprs)
                           .with_columns(pl.lit("total").alias("section"))
                )
                
                # Ensure same column order for concatenation
                # Reorder monthly_total to match monthly_by_section column order
                section_cols = ["gvkey","iid","month_end","country","continent","section"]
                other_cols = [c for c in monthly_by_section.columns if c not in section_cols]
                monthly_total = monthly_total.select([*section_cols, *other_cols])
                
                # Union: per-section + totals
                monthly = pl.concat([monthly_by_section, monthly_total], how="vertical_relaxed")
            else:
                # No valid sections, use standard aggregation
                monthly = (
                    df_rows.group_by("gvkey","iid","month_end","country","continent")
                           .agg(agg_exprs)
                           .with_columns(pl.lit(None).cast(pl.Utf8).alias("section"))
                )
        else:
            # Standard aggregation (non-USA or no section requirement)
            monthly = (
                df_rows.group_by("gvkey","iid","month_end","country","continent")
                       .agg(agg_exprs)
            )
            # Add section column for consistency (null for non-USA)
            if "section" not in monthly.columns:
                monthly = monthly.with_columns(pl.lit(None).cast(pl.Utf8).alias("section"))

        # Save per-shard checkpoint
        chk_dir = Path(checkpoint_dir)
        chk_dir.mkdir(parents=True, exist_ok=True)
        out_tmp = chk_dir / f"{p.stem}__monthly.parquet"
        monthly.rechunk().write_parquet(out_tmp)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return {"shard": shard_path, "status": "success", "tmp": str(out_tmp)}

    except Exception as e:
        traceback.print_exc()
        return {"shard": shard_path, "status": "error", "error": str(e)}

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Monthly FinBERT scoring (spawn-safe, manifest-ready)")
    ap.add_argument("--paths", default="config/paths.yaml")
    ap.add_argument("--device", default="cuda", choices=["cpu","cuda"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--lag_days", type=int, default=7)
    ap.add_argument("--num_workers", type=int, default=min(8, multiprocessing.cpu_count()))
    ap.add_argument("--fp16", action="store_true")

    # Inputs
    ap.add_argument("--manifest", default="")
    ap.add_argument("--shards_dir", default="")
    ap.add_argument("--scope", default="global", choices=["global","global_ex_us","country","continent"])
    ap.add_argument("--scope_value", default="")
    ap.add_argument("--exclude_path_regex", default="")
    ap.add_argument("--limit_shards", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")

    # Outputs
    ap.add_argument("--out", default="", help="Output file path for final results (default: from paths.yaml or finbert_master.parquet)")
    ap.add_argument("--checkpoint_dir", default="data/textsec/processed/FinBert/checkpoints", help="Directory for intermediate checkpoint files")
    ap.add_argument("--checkpoint_every", type=int, default=5, help="Write rolling checkpoint every N shards processed")

    args = ap.parse_args()
    paths = read_paths(args.paths)
    out_master = Path(args.out or paths.get("textsec", {}).get("finbert_master", "data/textsec/processed/FinBert/finbert_master.parquet"))
    out_master.parent.mkdir(parents=True, exist_ok=True)

    # ---------- helpers (no imports here; pl is imported at module top) ----------
    def _normalize_month_end(df: pl.DataFrame) -> pl.DataFrame:
        """
        Ensure df has a proper pl.Date 'month_end'. If present but null/invalid,
        rebuild from filing_date or year/month if available.
        """
        # If column exists, try to parse/standardize first
        if "month_end" in df.columns:
            s = df["month_end"]
            if s.dtype == pl.Datetime:
                df = df.with_columns(pl.col("month_end").dt.date())
            elif s.dtype == pl.Utf8:
                # try strict formats first
                for fmt in ("%Y-%m-%d", "%Y%m%d"):
                    df_try = df.with_columns(pl.col("month_end").str.strptime(pl.Date, fmt=fmt, strict=True))
                    if df_try["month_end"].null_count() == 0:
                        df = df_try
                        break
                else:
                    # YYYY-MM / YYYYMM → end of month
                    df_try = df.with_columns(
                        pl.col("month_end").str.strptime(pl.Date, fmt="%Y-%m", strict=True).dt.month_end()
                    )
                    if df_try["month_end"].null_count() == 0:
                        df = df_try
                    else:
                        df_try = df.with_columns(
                            pl.col("month_end").str.strptime(pl.Date, fmt="%Y%m", strict=True).dt.month_end()
                        )
                        if df_try["month_end"].null_count() == 0:
                            df = df_try
                        else:
                            df = df.with_columns(pl.col("month_end").str.strptime(pl.Date, strict=False))

            # If still all nulls, rebuild
            if df["month_end"].null_count() == len(df):
                return _rebuild_month_end_if_null(df)
            return df

        # No column → rebuild
        return _rebuild_month_end_if_null(df)

    def _rebuild_month_end_if_null(df: pl.DataFrame) -> pl.DataFrame:
        """
        Construct 'month_end' from available date info:
        1) 'filing_date' (datetime or string)
        2) integer/string year+month
        3) any common date-like columns
        """
        # 1) filing_date
        if "filing_date" in df.columns:
            s = df["filing_date"]
            if s.dtype == pl.Datetime:
                return df.with_columns(pl.col("filing_date").dt.date().dt.month_end().alias("month_end"))
            if s.dtype == pl.Date:
                return df.with_columns(pl.col("filing_date").dt.month_end().alias("month_end"))
            if s.dtype == pl.Utf8:
                for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
                    df_try = df.with_columns(pl.col("filing_date").str.strptime(pl.Date, fmt=fmt, strict=True))
                    if df_try["filing_date"].null_count() < len(df_try):
                        return df_try.with_columns(pl.col("filing_date").dt.month_end().alias("month_end"))
                df_try = df.with_columns(pl.col("filing_date").str.strptime(pl.Date, strict=False))
                if df_try["filing_date"].null_count() < len(df_try):
                    return df_try.with_columns(pl.col("filing_date").dt.month_end().alias("month_end"))

        # 2) year + month (ints/strings)
        if "year" in df.columns and "month" in df.columns:
            df_tmp = df.with_columns([
                pl.col("year").cast(pl.Int32, strict=False).alias("year__i"),
                pl.col("month").cast(pl.Int32, strict=False).alias("month__i"),
            ])
            if df_tmp["year__i"].null_count() < len(df_tmp) and df_tmp["month__i"].null_count() < len(df_tmp):
                return df_tmp.with_columns(
                    pl.date(pl.col("year__i"), pl.col("month__i"), pl.lit(1, dtype=pl.Int32)).dt.month_end().alias("month_end")
                )

        # 3) any other date-like columns
        for cand in ("report_date", "date", "doc_date", "period_end"):
            if cand in df.columns:
                s = df[cand]
                if s.dtype == pl.Datetime:
                    return df.with_columns(pl.col(cand).dt.date().dt.month_end().alias("month_end"))
                if s.dtype == pl.Date:
                    return df.with_columns(pl.col(cand).dt.month_end().alias("month_end"))
                if s.dtype == pl.Utf8:
                    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
                        df_try = df.with_columns(pl.col(cand).str.strptime(pl.Date, fmt=fmt, strict=True))
                        if df_try[cand].null_count() < len(df_try):
                            return df_try.with_columns(pl.col(cand).dt.month_end().alias("month_end"))
                    df_try = df.with_columns(pl.col(cand).str.strptime(pl.Date, strict=False))
                    if df_try[cand].null_count() < len(df_try):
                        return df_try.with_columns(pl.col(cand).dt.month_end().alias("month_end"))

        # give up: create a null Date column (so schema is correct)
        return df.with_columns(pl.lit(None, dtype=pl.Date).alias("month_end"))


    def _normalize_ids_and_types(df: pl.DataFrame) -> pl.DataFrame:
        # gvkey → try numeric cast first, then fall back, then make string
        if "gvkey" in df.columns:
            df = df.with_columns(
                pl.coalesce([
                    pl.col("gvkey").cast(pl.Int64, strict=False),
                    pl.col("gvkey"),
                ])
                .cast(pl.Utf8)
                .alias("gvkey")
            )

        if "iid" in df.columns:
            df = df.with_columns(pl.col("iid").cast(pl.Utf8).str.strip_chars())

        if "doc_sent_count" in df.columns:
            df = df.with_columns(pl.col("doc_sent_count").cast(pl.Int64))

        float_cols = [
            "prob_neg_mean","prob_neu_mean","prob_pos_mean",
            "sent_pos_minus_neg_mean",
        ]
        # Only add ratio columns if they exist
        for ratio_col in ["ratio_sent_pos_gt_neg", "ratio_sent_neg_gt_pos"]:
            if ratio_col in df.columns:
                float_cols.append(ratio_col)
        present = [c for c in float_cols if c in df.columns]
        if present:
            df = df.with_columns([pl.col(c).cast(pl.Float64) for c in present])

        for txt in ("company_name","country","continent","excntry"):
            if txt in df.columns:
                df = df.with_columns(pl.col(txt).cast(pl.Utf8))

        return df

    # ---------- build shard list ----------
    if args.manifest:
        man = _load_manifest(Path(args.manifest))
        man = _apply_exclude_regex(man, args.exclude_path_regex or "")
        man = _filter_scope(man, args.scope, args.scope_value)
        shard_paths = man["__shard_path"].to_list()
    elif args.shards_dir:
        shard_paths = [str(p) for p in _discover_shards(Path(args.shards_dir))]
        if args.scope == "global_ex_us":
            shard_paths = [p for p in shard_paths if "/USA/" not in p.replace("\\","/")]
        elif args.scope == "country" and args.scope_value:
            shard_paths = [p for p in shard_paths if f"/{args.scope_value}/" in p.replace("\\","/")]
    else:
        shard_paths = []

    if args.shuffle and len(shard_paths) > 1:
        import random; random.shuffle(shard_paths)
    if args.limit_shards and args.limit_shards > 0:
        shard_paths = shard_paths[:args.limit_shards]

    # Detect if we're processing USA scope
    is_usa_scope = (args.scope == "country" and args.scope_value.upper() in ("USA", "US", "UNITED STATES")) or \
                   (args.scope == "global" and "/USA/" in " ".join(shard_paths))
    
    if is_usa_scope:
        log(f"🇺🇸 USA scope detected: will aggregate by section + totals")

    log(f"🧭 Processing {len(shard_paths)} shard(s) with {args.num_workers} workers")
    log(f"🖥️  Device: {args.device}, Batch size: {args.batch_size}, CPU threads: {multiprocessing.cpu_count()}")

    temp_paths: List[str] = []
    processed = 0

    mp_ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_ctx) as ex:
        futs = {
            ex.submit(
                process_shard, sp, args.device, args.batch_size, args.max_length,
                args.lag_days, args.checkpoint_dir, args.fp16, is_usa_scope
            ): sp for sp in shard_paths
        }
        for fut in as_completed(futs):
            res = fut.result()
            shard_name = Path(res.get("shard","unknown")).name
            status = res.get("status","unknown")
            if status == "success":
                processed += 1
                tmp = res.get("tmp")
                if tmp: temp_paths.append(tmp)
                log(f"  ✅ [{processed}/{len(shard_paths)}] {shard_name}")
                if args.checkpoint_every and processed % max(1, args.checkpoint_every) == 0 and temp_paths:
                    try:
                        parts = [pl.read_parquet(p) for p in temp_paths[-min(len(temp_paths), 10):]]
                        parts = [ _normalize_ids_and_types(_normalize_month_end(df)) for df in parts ]
                        tmp_out = Path(args.checkpoint_dir) / "merged_checkpoint.parquet"
                        pl.concat(parts, how="vertical_relaxed").rechunk().write_parquet(tmp_out)
                        log(f"  💾 wrote rolling merged checkpoint → {tmp_out}")
                    except Exception as e:
                        log(f"  ⚠️  rolling checkpoint failed: {e}")
            elif status == "empty_after_filter":
                processed += 1
                log(f"  ⚠️  [{processed}/{len(shard_paths)}] {shard_name} :: empty after filter")
            else:
                log(f"  ❌ {shard_name} :: {status} - {res.get('error','')}")

    # ---------- Final merge (instance-level) ----------
    if not temp_paths:
        log("⚠️  No monthly results produced. Attempting to rebuild from checkpoints...")
        cands = sorted(str(p) for p in Path(args.checkpoint_dir).glob("*__monthly.parquet"))
        if not cands:
            log("⚠️  No checkpoints found. Aborting.")
            return 2
        parts = [pl.read_parquet(fp) for fp in cands]
    else:
        parts = [pl.read_parquet(fp) for fp in temp_paths]

    # Normalize before writing out the instance master
    parts = [ _normalize_ids_and_types(_normalize_month_end(df)) for df in parts ]
    final = pl.concat(parts, how="vertical_relaxed").rechunk()

    # If we have section column (USA scope), check for duplicates and aggregate if needed
    if "section" in final.columns:
        keys = ["gvkey", "iid", "month_end", "country", "continent", "section"]
        # Check for duplicates
        dup_cnt = (
            final.group_by(keys)
                 .len()
                 .filter(pl.col("len") > 1)
                 .height
        )
        if dup_cnt > 0:
            log(f"  📊 Aggregating {dup_cnt} duplicate (gvkey, iid, month_end, section) combinations")
            # Weighted aggregation for metrics
            wsum = lambda col: (pl.col(col) * pl.col("doc_sent_count")).sum()
            safe_div = lambda num, den: (num / pl.when(den == 0).then(None).otherwise(den))
            
            agg_exprs = [pl.sum("doc_sent_count").alias("doc_sent_count")]
            if "prob_neg_mean" in final.columns:
                agg_exprs.append(safe_div(wsum("prob_neg_mean"), pl.sum("doc_sent_count")).alias("prob_neg_mean"))
            if "prob_neu_mean" in final.columns:
                agg_exprs.append(safe_div(wsum("prob_neu_mean"), pl.sum("doc_sent_count")).alias("prob_neu_mean"))
            if "prob_pos_mean" in final.columns:
                agg_exprs.append(safe_div(wsum("prob_pos_mean"), pl.sum("doc_sent_count")).alias("prob_pos_mean"))
            if "sent_pos_minus_neg_mean" in final.columns:
                agg_exprs.append(safe_div(wsum("sent_pos_minus_neg_mean"), pl.sum("doc_sent_count")).alias("sent_pos_minus_neg_mean"))
            if "ratio_sent_pos_gt_neg" in final.columns:
                agg_exprs.append(safe_div(wsum("ratio_sent_pos_gt_neg"), pl.sum("doc_sent_count")).alias("ratio_sent_pos_gt_neg"))
            if "ratio_sent_neg_gt_pos" in final.columns:
                agg_exprs.append(safe_div(wsum("ratio_sent_neg_gt_pos"), pl.sum("doc_sent_count")).alias("ratio_sent_neg_gt_pos"))
            agg_exprs.extend([
                pl.first("company_name").alias("company_name"),
                pl.first("excntry").alias("excntry"),
                pl.first("section").alias("section"),
            ])
            
            final = (
                final.group_by(keys)
                     .agg(agg_exprs)
                     .sort("month_end", "country", "gvkey", "iid", "section")
            )

    # Compute SEC sentiment features on the merged monthly dataset
    try:
        final = _compute_sec_features(final)
    except Exception as e:
        log(f"⚠️  feature computation failed: {e}")

    # quick sanity log
    try:
        info = final.select(
            pl.col("month_end").min().alias("min_month"),
            pl.col("month_end").max().alias("max_month"),
            pl.col("month_end").n_unique().alias("n_months"),
            pl.len().alias("rows")
        ).to_dicts()[0]
        log(f"🧪 month_end sanity → min={info['min_month']} max={info['max_month']} unique={info['n_months']:,} rows={info['rows']:,}")
    except Exception as _:
        pass

    final.write_parquet(out_master)
    log(f"✅ FinBERT master scores saved → {out_master}")
    log("🎉 Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

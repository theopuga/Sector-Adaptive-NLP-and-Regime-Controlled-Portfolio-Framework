"""
Required columns (case-insensitive):
  date, exdate, cp_flag, strike_price, best_bid, best_offer,
  volume, open_interest, impl_volatility, delta, vega, forward_price,
  exercise_style
And one identifier among: ticker | underlying_symbol | secid | symbol | underlying

This script processes options data from two sources:
  1. Primary file (default: etfs_options_data.gz) - filtered for SPX by default
  2. Secondary file (default: spy_options_2023_09.csv.gz) - used when primary runs out
     - Filters for SPY and maps it to SPX
     - Starts from the day after primary file ends

Example:
  python build_opts_features_new.py \
    --input ../assets/etfs_options_data.gz \
    --input2 ../assets/spy_options_2023_09.csv.gz \
    --out_monthly ./test/opts_sentiment_monthly.csv \
    --ticker SPX \
    --ticker2 SPY \
    --end_date 2025-06-30 \
    --short_dte 10 --short_dte_max 60 --long_dte 270 \
    --zscore_mode expanding --train_end 2014-12-31 --min_periods 12 \
    --chunksize 2_000_000
"""

import argparse
import numpy as np
import pandas as pd

REQ_COLS = [
    "date","exdate","cp_flag","strike_price","best_bid","best_offer",
    "volume","open_interest","impl_volatility","delta","vega","forward_price","exercise_style"
]
ID_CANDIDATES = ["ticker","underlying_symbol","secid","symbol","underlying","act_symbol"]

# ---------- Utils ----------

def lower_cols(df): return df.rename(columns={c: c.lower() for c in df.columns})

def find_id_col(cols):
    cols = [c.lower() for c in cols]
    for c in ID_CANDIDATES:
        if c in cols: return c
    raise ValueError(f"No identifier column found. Expected one of {ID_CANDIDATES}")

def ensure_style_col(df):
    if "exercise_style" not in df.columns:
        df["exercise_style"] = "UNKNOWN"
    else:
        df["exercise_style"] = df["exercise_style"].astype("string").str.upper().str.strip().fillna("UNKNOWN")
    return df

def parse_ymd_series(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s_digits = s.str.replace(r"\D", "", regex=True)
    out = pd.to_datetime(s_digits, format="%Y%m%d", errors="coerce")
    return out.fillna(pd.to_datetime(s, errors="coerce"))

def safe_div(a, b):
    b = b.replace({0: np.nan})
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)

def map_dolt_schema_to_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Dolt schema columns to standard OptionMetrics-style columns.
    Dolt schema: date, act_symbol, expiration, strike, call_put, bid, ask, vol, delta, gamma, theta, vega, rho
    Standard: date, exdate, cp_flag, strike_price, best_bid, best_offer, volume, open_interest, 
              impl_volatility, delta, vega, forward_price, exercise_style
    """
    df = df.copy()
    df_lower = df.rename(columns={c: c.lower() for c in df.columns})
    
    # Map columns
    # Note: "vol" in Dolt schema might be implied volatility, not volume
    # Check if "vol" values are in typical IV range (0-2) vs volume (much larger)
    is_iv = False
    if "vol" in df_lower.columns:
        vol_sample = pd.to_numeric(df_lower["vol"].head(100), errors="coerce").dropna()
        if len(vol_sample) > 0:
            # If most values are < 2, it's likely IV; if > 10, it's likely volume
            median_vol = vol_sample.median()
            is_iv = median_vol < 2.0
    
    column_mapping = {
        "expiration": "exdate",
        "strike": "strike_price",
        "call_put": "cp_flag",
        "bid": "best_bid",
        "ask": "best_offer"
    }
    
    # Handle "vol" specially based on whether it's IV or volume
    if "vol" in df_lower.columns:
        if is_iv:
            df_lower["impl_volatility"] = pd.to_numeric(df_lower["vol"], errors="coerce")
            df_lower["volume"] = 0.0  # Default volume to 0 if vol is actually IV
        else:
            df_lower["volume"] = pd.to_numeric(df_lower["vol"], errors="coerce")
    
    # Rename other columns
    for old_col, new_col in column_mapping.items():
        if old_col in df_lower.columns:
            df_lower = df_lower.rename(columns={old_col: new_col})
    
    # Ensure numeric columns
    numeric_cols = ["best_bid", "best_offer", "strike_price", "volume", "delta", "vega"]
    for col in numeric_cols:
        if col in df_lower.columns:
            df_lower[col] = pd.to_numeric(df_lower[col], errors="coerce")
    
    # Compute mid price
    if "best_bid" in df_lower.columns and "best_offer" in df_lower.columns:
        df_lower["mid"] = (df_lower["best_bid"] + df_lower["best_offer"]) / 2.0
    else:
        df_lower["mid"] = np.nan
    
    # Add missing required columns with defaults/NaN
    if "volume" not in df_lower.columns:
        df_lower["volume"] = 0.0  # Default to 0 if not available
    
    if "open_interest" not in df_lower.columns:
        df_lower["open_interest"] = 0.0  # Default to 0 if not available
    
    # Try to find implied volatility in various forms
    if "impl_volatility" not in df_lower.columns:
        # Check for alternative names
        alt_iv_names = ["iv", "implied_vol", "implied_volatility", "sigma"]
        found_iv = False
        for alt_name in alt_iv_names:
            if alt_name in df_lower.columns:
                df_lower["impl_volatility"] = pd.to_numeric(df_lower[alt_name], errors="coerce")
                found_iv = True
                break
        if not found_iv:
            # If still not found, try to infer from option pricing (very approximate)
            # This is a fallback - will likely be filtered out in processing
            df_lower["impl_volatility"] = np.nan
    
    # Compute forward_price from put-call parity if possible
    if "forward_price" not in df_lower.columns:
        df_lower["forward_price"] = np.nan
        # Try to compute from ATM options using put-call parity approximation
        # For ATM: F ≈ Strike + (C_mid - P_mid), simplified
        if "strike_price" in df_lower.columns and "mid" in df_lower.columns:
            # Group by date, expiration, strike to match calls and puts
            # This is approximate - would need proper matching
            pass  # Leave as NaN for now - will be handled in processing
    
    if "exercise_style" not in df_lower.columns:
        df_lower["exercise_style"] = "UNKNOWN"
    
    return df_lower

# ---------- Chunk processing: build partial SUMS/COUNTS per (date, id, style) ----------

def process_chunk(df: pd.DataFrame, short_dte, short_dte_max, long_dte) -> pd.DataFrame:
    df = lower_cols(df)
    id_col = find_id_col(df.columns)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = ensure_style_col(df)

    # dates
    df["date"]   = parse_ymd_series(df["date"])
    df["exdate"] = parse_ymd_series(df["exdate"])
    df = df.dropna(subset=["date","exdate","cp_flag","impl_volatility", id_col])

    # numeric casts
    num_cols = ["strike_price","best_bid","best_offer","volume","open_interest",
                "impl_volatility","delta","vega","forward_price"]
    for c in num_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # derived
    df["mid"] = (df["best_bid"] + df["best_offer"]) / 2.0
    df["moneyness"] = (df["strike_price"] / df["forward_price"]).replace([np.inf, -np.inf], np.nan)
    df["ttm_days"] = (df["exdate"] - df["date"]).dt.days
    df["cp_flag"] = df["cp_flag"].astype("string").str.upper().str.strip()

    # masks
    is_put  = df["cp_flag"].eq("P")
    is_call = df["cp_flag"].eq("C")
    atm     = df["moneyness"].between(0.95, 1.05, inclusive="neither")
    shortT  = df["ttm_days"].between(short_dte, short_dte_max, inclusive="both")
    longT   = df["ttm_days"] >= long_dte
    put25   = is_put  & (df["delta"] < 0.25)
    call75  = is_call & (df["delta"] > 0.75)

    df["vega_oi"] = df["vega"] * df["open_interest"]

    idx = ["date", id_col, "exercise_style"]

    # Put/Call sums
    pc_vol = df.pivot_table(index=idx, columns="cp_flag", values="volume", aggfunc="sum")
    pc_oi  = df.pivot_table(index=idx, columns="cp_flag", values="open_interest", aggfunc="sum")
    vo     = df.pivot_table(index=idx, columns="cp_flag", values="vega_oi", aggfunc="sum")

    # helper to extract from pivot safely
    def get_pivot_col(pvt, key):
        if isinstance(pvt.columns, pd.MultiIndex):
            if (key,) in pvt.columns:
                return pvt[(key,)]
            return pd.Series(index=pvt.index, dtype=float)
        else:
            if key in pvt.columns:
                return pvt[key]
            return pd.Series(index=pvt.index, dtype=float)

    base = pd.DataFrame(index=pc_vol.index.union(pc_oi.index).union(vo.index)).sort_index()
    base["sum_vol_put"]    = get_pivot_col(pc_vol, "P")
    base["sum_vol_call"]   = get_pivot_col(pc_vol, "C")
    base["sum_oi_put"]     = get_pivot_col(pc_oi,  "P")
    base["sum_oi_call"]    = get_pivot_col(pc_oi,  "C")
    base["sum_vegaoi_put"] = get_pivot_col(vo,     "P")
    base["sum_vegaoi_call"]= get_pivot_col(vo,     "C")

    # totals for weights
    base["oi_total"]  = base["sum_oi_put"].fillna(0)  + base["sum_oi_call"].fillna(0)
    base["vol_total"] = base["sum_vol_put"].fillna(0) + base["sum_vol_call"].fillna(0)

    # Sums & counts for ATM short/long and skew buckets
    def sum_cnt(mask):
        g = df.loc[mask, ["impl_volatility"] + idx].groupby(idx)["impl_volatility"]
        return g.sum(), g.count()

    iv_short_sum, iv_short_cnt = sum_cnt(atm & shortT)
    iv_long_sum,  iv_long_cnt  = sum_cnt(longT & atm)
    iv_put25_sum, iv_put25_cnt = sum_cnt(put25)
    iv_call75_sum, iv_call75_cnt = sum_cnt(call75)

    base["iv_short_sum"] = iv_short_sum
    base["iv_short_cnt"] = iv_short_cnt
    base["iv_long_sum"]  = iv_long_sum
    base["iv_long_cnt"]  = iv_long_cnt
    base["iv_put25_sum"] = iv_put25_sum
    base["iv_put25_cnt"] = iv_put25_cnt
    base["iv_call75_sum"]= iv_call75_sum
    base["iv_call75_cnt"]= iv_call75_cnt

    base.index.names = ["date", id_col, "exercise_style"]
    return base

# ---------- Finalize daily to features per (date, id, style) ----------

def finalize_daily(agg: pd.DataFrame) -> pd.DataFrame:
    df = agg.sort_index().copy()

    df["putcall_vol"] = safe_div(df["sum_vol_put"],  df["sum_vol_call"])
    df["putcall_oi"]  = safe_div(df["sum_oi_put"],   df["sum_oi_call"])
    df["vegaoi_tilt"] = safe_div(df["sum_vegaoi_put"], df["sum_vegaoi_call"])

    iv_short = safe_div(df["iv_short_sum"], df["iv_short_cnt"])
    iv_long  = safe_div(df["iv_long_sum"],  df["iv_long_cnt"])
    df["iv_term_slope"] = iv_long - iv_short  # long − short

    iv_put25  = safe_div(df["iv_put25_sum"],  df["iv_put25_cnt"])
    iv_call75 = safe_div(df["iv_call75_sum"], df["iv_call75_cnt"])
    df["iv_skew"] = iv_put25 - iv_call75

    keep = [
        "putcall_vol","putcall_oi","vegaoi_tilt","iv_term_slope","iv_skew",
        "oi_total","vol_total"
    ]
    return df[keep]

# ---------- Z-scores (STRICTLY CAUSAL) ----------

def causal_expanding_zscore(s: pd.Series, min_periods: int = 12) -> pd.Series:
    hist = s.shift(1)
    mu = hist.expanding(min_periods=min_periods).mean()
    sd = hist.expanding(min_periods=min_periods).std(ddof=0)
    z = (s - mu) / sd
    z[sd==0] = np.nan
    return z

def insample_zscore_over_mask(s: pd.Series, mask: pd.Series) -> pd.Series:
    sub = s[mask]
    mu, sd = sub.mean(), sub.std(ddof=0)
    out = pd.Series(index=s.index, dtype=float)
    if sd and np.isfinite(sd) and sd != 0:
        out.loc[mask] = (sub - mu) / sd
    else:
        out.loc[mask] = np.nan
    return out

# ---------- Monthly per (ticker, style) ----------

def build_monthly_per_style(daily: pd.DataFrame, zmode: str, train_end: str, min_periods: int) -> pd.DataFrame:
    # daily index: (date, id, style)
    id_name, style_name = daily.index.names[1], daily.index.names[2]
    rows = []

    # We'll aggregate monthly OI/Volume weights by SUM over that month (more stable)
    for (id_val, style_val), sub in daily.groupby(level=[1,2], sort=True):
        s = sub.copy()
        s.index = s.index.droplevel([1,2])  # keep date only
        s = s.sort_index()

        monthly_raw = pd.DataFrame(index=s.resample("D").asfreq().index)  # ensure continuous before resample
        # Use daily values directly (already features). For weights, we sum within month.
        monthly_last = s.resample("ME").last()
        monthly_sum  = s[["oi_total","vol_total"]].resample("ME").sum(min_count=1)

        monthly = monthly_last.join(monthly_sum[["oi_total","vol_total"]], rsuffix="_sum")

        cutoff = pd.to_datetime(train_end)
        mask_train = monthly.index <= cutoff
        mask_live  = monthly.index >  cutoff

        def hybrid_z(col, invert=False):
            x = monthly[col]
            if zmode == "expanding":
                z = causal_expanding_zscore(x, min_periods=min_periods)
            else:  # 'hybrid' default
                z_train = insample_zscore_over_mask(x, mask_train)
                z_live  = causal_expanding_zscore(x, min_periods=min_periods)
                z = z_train.combine_first(z_live)
                z.loc[mask_live] = z_live.loc[mask_live]
            return -z if invert else z

        z = pd.DataFrame(index=monthly.index)
        z["z_putcall_vol"] = hybrid_z("putcall_vol")                 # higher = bearish
        z["z_putcall_oi"]  = hybrid_z("putcall_oi")                  # higher = bearish
        z["z_vegaoi_tilt"] = hybrid_z("vegaoi_tilt")                 # higher = bearish
        z["z_iv_term"]     = hybrid_z("iv_term_slope", invert=True)  # flatter/inverted = bearish
        z["z_iv_skew"]     = hybrid_z("iv_skew")                     # higher = bearish
        z["SSI_options"]   = z.mean(axis=1)

        m = monthly.join(z)
        m["id"] = id_val                      # <- normalize column name
        m[style_name] = style_val
        rows.append(m.reset_index().rename(columns={"index":"date"}))

    if not rows:
        return pd.DataFrame(columns=["date", "id", style_name])

    monthly_all = pd.concat(rows, ignore_index=True)
    # rename weight columns for clarity
    if "oi_total_sum" not in monthly_all.columns:
        monthly_all = monthly_all.rename(columns={"oi_total":"oi_total_sum","vol_total":"vol_total_sum"})
    # ensure lowercase 'date' already set; 'id' already created above
    return monthly_all

# ---------- Blend styles → per ticker monthly ----------

BLEND_RAW_COLS = ["putcall_vol","putcall_oi","vegaoi_tilt","iv_term_slope","iv_skew"]
BLEND_Z_COLS   = ["z_putcall_vol","z_putcall_oi","z_vegaoi_tilt","z_iv_term","z_iv_skew","SSI_options"]

def blend_styles(monthly_style_df: pd.DataFrame) -> pd.DataFrame:
    if monthly_style_df.empty:
        return monthly_style_df

    # Ensure there is a normalized 'id' column to use downstream
    if "id" not in monthly_style_df.columns:
        # find original identifier and mirror into 'id' without changing anything else
        orig_id = next((c for c in monthly_style_df.columns if c in ID_CANDIDATES), None)
        if orig_id is None:
            raise ValueError("No identifier column found in monthly_style_df.")
        monthly_style_df = monthly_style_df.copy()
        monthly_style_df["id"] = monthly_style_df[orig_id]

    style_name = "exercise_style"

    # weight = monthly OI sum; fallback to volume sum; else equal
    def compute_weights(df):
        w = df["oi_total_sum"].astype(float)
        if w.isna().all() or (w.fillna(0).sum() == 0):
            w = df["vol_total_sum"].astype(float)
        if w.isna().all() or (w.fillna(0).sum() == 0):
            w = pd.Series(1.0, index=df.index)
        return w

    out_rows = []
    for (dt, sid), grp in monthly_style_df.groupby(["date", "id"], sort=True):
        w = compute_weights(grp).fillna(0.0)
        wsum = w.sum()
        if wsum == 0:
            w = pd.Series(1.0, index=grp.index)
            wsum = float(len(w))
        w = w / wsum

        combined = {"date": dt, "id": sid}
        for col in BLEND_RAW_COLS + BLEND_Z_COLS:
            if col in grp.columns:
                combined[col] = np.nansum(w.values * grp[col].values)
            else:
                combined[col] = np.nan

        out_rows.append(combined)

    combined_df = pd.DataFrame(out_rows).sort_values(["date", "id"])
    cols = ["date", "id"] + BLEND_RAW_COLS + BLEND_Z_COLS
    return combined_df[cols]

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../assets/etfs_options_data.gz", help="Path to CSV(.gz) with many tickers & exercise_style.")
    ap.add_argument("--input2", default="../assets/spy_options_2023_09.csv.gz", help="Path to secondary CSV(.gz) file (e.g., SPY data to complement SPX).")
    ap.add_argument("--out_monthly", default="./opts_sentiment_monthly.csv", help="Output CSV path.")
    ap.add_argument("--ticker", default="SPX", help="Filter for specific ticker only (default: SPX). Set to SPY for secondary file.")
    ap.add_argument("--ticker2", default="SPY", help="Ticker name in secondary file (default: SPY, will be mapped to primary ticker).")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--end_date", default="2025-06-30", help="End date for data processing (YYYY-MM-DD).")

    # DTE (calendar days)
    ap.add_argument("--short_dte", type=int, default=10, help="Short-end min DTE (inclusive).")
    ap.add_argument("--short_dte_max", type=int, default=60, help="Short-end max DTE (inclusive).")
    ap.add_argument("--long_dte", type=int, default=270, help="Long-end min DTE for term slope.")

    # z-scores
    ap.add_argument("--zscore_mode", choices=["hybrid","expanding"], default="hybrid",
                    help="Hybrid: in-sample z for <=train_end, expanding causal after; Expanding: all expanding.")
    ap.add_argument("--train_end", default="2014-12-31", help="YYYY-MM-DD cutoff for 'hybrid' mode.")
    ap.add_argument("--min_periods", type=int, default=12, help="Min history for expanding z-scores.")
    args = ap.parse_args()

    end_date = pd.to_datetime(args.end_date)
    target_ticker = args.ticker.upper()
    
    partials = []
    
    # ========== Process first file (primary source) ==========
    print(f"[INFO] Reading {args.input} in chunks of {args.chunksize} …")
    print(f"[INFO] Filtering for ticker: {target_ticker}")
    
    last_date_primary = None
    reader = pd.read_csv(
        args.input,
        chunksize=args.chunksize,
        dtype={"date":"string","exdate":"string"},
        low_memory=False
    )

    for i, chunk in enumerate(reader, 1):
        # Filter for specific ticker (use lowercase for consistency)
        chunk_lower = chunk.rename(columns={c: c.lower() for c in chunk.columns})
        id_col = find_id_col(chunk.columns)
        id_col_lower = id_col.lower()
        if id_col_lower in chunk_lower.columns:
            chunk_lower = chunk_lower[chunk_lower[id_col_lower].str.upper() == target_ticker]
            if len(chunk_lower) == 0:
                print(f"[INFO] chunk {i}: no data for ticker {target_ticker}")
                continue
            chunk = chunk_lower
        
        # Filter to end date
        chunk["date"] = parse_ymd_series(chunk["date"])
        chunk = chunk[chunk["date"] <= end_date]
        if len(chunk) == 0:
            continue
        
        p = process_chunk(chunk, args.short_dte, args.short_dte_max, args.long_dte)
        partials.append(p)
        if len(p):
            idx = p.index
            first = idx.get_level_values(0).min()
            last  = idx.get_level_values(0).max()
            last_date_primary = max(last_date_primary, last) if last_date_primary is not None else last
            print(f"[INFO] chunk {i}: {first} → {last}  ({len(p)} rows (date,id,style))")
        else:
            print(f"[INFO] chunk {i}: no valid rows")

    # ========== Process second file (fallback source) ==========
    if last_date_primary is not None and args.input2:
        print(f"\n[INFO] Primary file ends at: {last_date_primary}")
        print(f"[INFO] Reading {args.input2} starting from {last_date_primary} …")
        print(f"[INFO] Filtering for ticker: {args.ticker2} (will map to {target_ticker})")
        
        start_date_secondary = last_date_primary + pd.Timedelta(days=1)
        
        reader2 = pd.read_csv(
            args.input2,
            chunksize=args.chunksize,
            dtype={"date":"string","exdate":"string"},
            low_memory=False
        )

        for i, chunk in enumerate(reader2, 1):
            # First, map Dolt schema to standard format
            chunk = map_dolt_schema_to_standard(chunk)
            
            # Filter for ticker in secondary file
            id_col = find_id_col(chunk.columns)
            if id_col in chunk.columns:
                chunk = chunk[chunk[id_col].str.upper() == args.ticker2.upper()]
                if len(chunk) == 0:
                    print(f"[INFO] chunk {i}: no data for ticker {args.ticker2}")
                    continue
            
            # Filter to date range and end date
            chunk["date"] = parse_ymd_series(chunk["date"])
            chunk = chunk[(chunk["date"] >= start_date_secondary) & (chunk["date"] <= end_date)]
            if len(chunk) == 0:
                continue
            
            # Map secondary ticker to primary ticker
            if id_col in chunk.columns:
                chunk[id_col] = target_ticker
            
            p = process_chunk(chunk, args.short_dte, args.short_dte_max, args.long_dte)
            partials.append(p)
            if len(p):
                idx = p.index
                first = idx.get_level_values(0).min()
                last  = idx.get_level_values(0).max()
                print(f"[INFO] chunk {i} (secondary): {first} → {last}  ({len(p)} rows (date,id,style))")
            else:
                print(f"[INFO] chunk {i} (secondary): no valid rows")

    if not partials:
        print("[WARN] No valid data found; writing empty file.")
        pd.DataFrame(columns=["date","id"]).to_csv(args.out_monthly, index=False)  # <- normalized headers
        return

    print("\n[INFO] Concatenating & summing partials …")
    agg = pd.concat(partials).groupby(level=[0,1,2]).sum(min_count=1).sort_index()

    print("[INFO] Finalizing DAILY features per (date, id, style) …")
    daily = finalize_daily(agg)

    print("[INFO] Building MONTHLY per-style features with strictly-causal z-scores …")
    monthly_style = build_monthly_per_style(daily, args.zscore_mode, args.train_end, args.min_periods)

    print("[INFO] Blending styles → per-ticker monthly (OI-weighted; fallback vol; else equal) …")
    monthly_combined = blend_styles(monthly_style)

    # ensure normalized column names on output
    monthly_combined = monthly_combined.rename(columns={"Date":"date"})  # just in case

    # Filter to end date
    monthly_combined["date"] = pd.to_datetime(monthly_combined["date"])
    monthly_combined = monthly_combined[monthly_combined["date"] <= end_date]
    monthly_combined = monthly_combined.sort_values("date")

    monthly_combined.to_csv(args.out_monthly, index=False)
    print(f"[OK] wrote {args.out_monthly}  (rows={len(monthly_combined)}, date range: {monthly_combined['date'].min()} to {monthly_combined['date'].max()})")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Map New Words to Sectors
========================
Purpose:
    This script assigns **newly emerging words** (not in the frozen dictionary) to sectors  
    based on correlation patterns between each word’s frequency time series and existing  
    sector-level term-frequency trends.

Inputs:
    --panel : Word trend parquet (`word_trends_monthly.parquet`)
    --sector_freq : Sector frequency parquet (`sector_word_frequency_monthly.parquet`)
    --frozen_dict : JSON dictionary frozen by `freeze_sector_dictionary.py`
    --restrict_to_oos : If set, restrict mapping to post-cutoff ("out-of-sample") words
    --n_jobs : Parallel worker count
    --min_points : Minimum non-zero months for correlation
    --checkpoint_every : Interval for progress saving
    --out_dir : Output folder for progressive mapping results

Outputs:
    - `new_words_sector_mapping_fixed12m.csv`  
      (each new word’s assigned sector, correlation metrics, ambiguity flag, etc.)

Core Logic:
    1. Loads all word-level and sector-level time series.
    2. Identifies new (unseen) words relative to frozen dictionary.
    3. For each new word, extracts its first 12 months of data after first appearance.
    4. Computes Pearson and Cosine similarities with sector time series.
    5. Assigns each word to the most correlated sector, flagging ambiguous cases.
    6. Runs in parallel batches for scalability.

Use Case:
    Expands sector vocabularies dynamically in a **leak-free, out-of-sample** fashion.

# Map new words to sectors command
python run/map_new_words_to_sectors.py \
  --panel data/textsec/processed/word_trends_monthly.parquet \
  --sector_freq data/textsec/processed/StaticDict/sector_word_frequency_monthly.parquet \
  --frozen_dict data/textsec/processed/FreezeDict/sector_dictionary.json \
  --restrict_to_oos --n_jobs 12 --min_points 9 \
  --checkpoint_every 50000 --out_dir data/textsec/processed/ProgressiveDict
"""
import argparse, json, time, sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime

def log(msg):
    ts = datetime.now().strftime("[%H:%M:%S]")
    print(f"{ts} {msg}")
    sys.stdout.flush()

def load_panel(fp: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(fp)
    need = {"gram","date","tf_share"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Panel missing columns: {miss}")
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date","gram","tf_share"]].copy()
    return df

def load_sector_series(fp: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(fp)
    need = {"date","sector","sector_tf_share"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Sector frequency file missing columns: {miss}")
    df["date"] = pd.to_datetime(df["date"])
    return df[["date","sector","sector_tf_share"]]

def load_frozen_dict(fp: str | Path) -> set[str]:
    d = json.loads(Path(fp).read_text())
    # words are values across all sectors
    return set(w for vs in d.values() for w in vs)

def first_seen_by_word(panel: pd.DataFrame) -> pd.Series:
    return panel.groupby("gram", sort=False)["date"].min()

def _pearson(a: pd.Series, b: pd.Series) -> float:
    a, b = a.astype("float64"), b.astype("float64")
    if a.count() < 4 or b.count() < 4:
        return np.nan
    try:
        return a.corr(b)
    except Exception:
        return np.nan

def _cosine(a: pd.Series, b: pd.Series) -> float:
    a = a.fillna(0.0).astype("float64")
    b = b.fillna(0.0).astype("float64")
    na = np.linalg.norm(a.values)
    nb = np.linalg.norm(b.values)
    if na == 0 or nb == 0: return np.nan
    return float(np.dot(a.values, b.values) / (na*nb))

def choose_sector_for_word(word_ts: pd.Series, sector_wide: pd.DataFrame) -> dict:
    # sector_wide: index=date, columns=sectors (sector_tf_share)
    # align & score
    scores = []
    for sector in sector_wide.columns:
        s = sector_wide[sector]
        # align
        join = pd.concat([word_ts.rename("w"), s.rename("s")], axis=1).dropna()
        if len(join) < 6:
            pear, cos = np.nan, np.nan
        else:
            pear = _pearson(join["w"], join["s"])
            cos  = _cosine(join["w"], join["s"])
        scores.append((sector, pear, cos))
    # rank by pearson then cosine
    scores = sorted(scores, key=lambda x: (np.nan_to_num(x[1], nan=-9), np.nan_to_num(x[2], nan=-9)), reverse=True)
    best = scores[0] if scores else (None, np.nan, np.nan)
    second = scores[1] if len(scores) > 1 else (None, np.nan, np.nan)
    margin = (best[1] - second[1]) if (not np.isnan(best[1]) and not np.isnan(second[1])) else np.nan
    # flag ambiguity: low corr or small margin
    ambiguous = (np.isnan(best[1]) or best[1] < 0.2 or (not np.isnan(margin) and margin < 0.05))
    return {
        "best_sector": best[0],
        "pearson": best[1],
        "cosine": best[2],
        "second_sector": second[0],
        "second_pearson": second[1],
        "margin_pearson": margin,
        "ambiguous": bool(ambiguous),
        "all_scores": scores
    }

def build_sector_wide(sector_df: pd.DataFrame) -> pd.DataFrame:
    wide = sector_df.pivot(index="date", columns="sector", values="sector_tf_share").sort_index()
    return wide

def fixed_after_12m_mapping(panel, sector_wide, new_words, first_seen):
    rows = []
    for w in sorted(new_words):
        fs = first_seen[w]
        # calibration window: first 12 months starting at first_seen
        end = (fs.to_period("M") + 11).to_timestamp()
        w_ts = (panel.loc[(panel["gram"]==w) & (panel["date"]>=fs) & (panel["date"]<=end), ["date","tf_share"]]
                      .set_index("date")["tf_share"])
        if w_ts.empty:
            # no usable series in first year; skip or mark ambiguous
            rows.append({
                "word": w, "first_seen": fs, "method": "fixed_after_12m",
                "assigned_sector": None, "pearson": np.nan, "cosine": np.nan,
                "second_sector": None, "second_pearson": np.nan, "margin_pearson": np.nan,
                "ambiguous": True
            })
            continue
        # restrict sector series to same window
        sector_slice = sector_wide.loc[w_ts.index.min(): w_ts.index.max()]
        res = choose_sector_for_word(w_ts, sector_slice)
        rows.append({
            "word": w, "first_seen": fs, "method": "fixed_after_12m",
            "assigned_sector": res["best_sector"], "pearson": res["pearson"], "cosine": res["cosine"],
            "second_sector": res["second_sector"], "second_pearson": res["second_pearson"],
            "margin_pearson": res["margin_pearson"], "ambiguous": res["ambiguous"]
        })
    return pd.DataFrame(rows)

def rolling_mapping(panel, sector_wide, new_words, first_seen, window_months=12):
    """Time-varying mapping per month (heavy)."""
    out = []
    for w in sorted(new_words):
        fs = first_seen[w]
        w_df = panel.loc[(panel["gram"]==w) & (panel["date"]>=fs), ["date","tf_share"]].set_index("date").sort_index()
        if w_df.empty: 
            continue
        # For each month t, use window [t-window+1, t]
        dates = w_df.index
        for t in dates:
            start = (t.to_period("M") - (window_months-1)).to_timestamp()
            w_win = w_df.loc[start:t]["tf_share"]
            sector_slice = sector_wide.loc[start:t]
            if len(w_win) < max(6, window_months//2):  # require minimum points
                continue
            res = choose_sector_for_word(w_win, sector_slice)
            out.append({
                "word": w, "date": t, "first_seen": fs, "method": f"rolling_{window_months}m",
                "assigned_sector": res["best_sector"], "pearson": res["pearson"], "cosine": res["cosine"],
                "second_sector": res["second_sector"], "second_pearson": res["second_pearson"],
                "margin_pearson": res["margin_pearson"], "ambiguous": res["ambiguous"]
            })
    return pd.DataFrame(out)

# === globals for workers (set by init_worker) ===
PANEL_12 = None          # pandas DataFrame with columns: gram, month_id, rel, tf_share
FS_IDX = None            # dict: word -> first_seen month_id (int)
SECTOR_ARR = None        # numpy array [n_months, n_sectors]
SECTORS = None           # list[str] sector names, aligned to SECTOR_ARR columns
MIN_M = None             # global min month_id (int)
PROCESSED_WORDS = set()  # words already in checkpoint (optional resume)

def init_worker(panel_12, fs_idx, sector_arr, sectors, min_m, processed_words):
    """Initializer runs once per process; saves large read-only objects into globals."""
    global PANEL_12, FS_IDX, SECTOR_ARR, SECTORS, MIN_M, PROCESSED_WORDS
    PANEL_12 = panel_12
    FS_IDX = fs_idx
    SECTOR_ARR = sector_arr
    SECTORS = sectors
    MIN_M = min_m
    PROCESSED_WORDS = processed_words

def _pearson(a, b):
    import numpy as np
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 6: return float("nan")
    aa = a[mask]; bb = b[mask]
    sa = aa.std(); sb = bb.std()
    if sa == 0 or sb == 0: return float("nan")
    return float(((aa - aa.mean()) * (bb - bb.mean())).mean() / (sa * sb))

def _cosine(a, b):
    import numpy as np
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0: return float("nan")
    aa = a[mask]; bb = b[mask]
    na = float((aa**2).sum() ** 0.5); nb = float((bb**2).sum() ** 0.5)
    if na == 0 or nb == 0: return float("nan")
    return float((aa * bb).sum() / (na * nb))

def process_batch(words):
    """Worker: map a batch of words -> rows (list[dict]). Uses globals set by init_worker."""
    import numpy as np
    out = []
    sub = PANEL_12[PANEL_12["gram"].isin(words)].copy()
    for w in words:
        if w in PROCESSED_WORDS:
            continue
        fs_m = int(FS_IDX[w])
        start = fs_m - MIN_M
        w_rows = sub[sub["gram"] == w]
        if w_rows.empty or start < 0 or (start + 12) > SECTOR_ARR.shape[0]:
            out.append({
                "word": w, "first_seen": fs_m, "method": "fixed_after_12m",
                "assigned_sector": None, "pearson": np.nan, "cosine": np.nan,
                "second_sector": None, "second_pearson": np.nan,
                "margin_pearson": np.nan, "ambiguous": True
            })
            continue
        # dense 12-vector for the word (0..11 relative months)
        w_vec = np.zeros(12, dtype="float32")
        rel_idx = w_rows["rel"].to_numpy(dtype="int64")
        w_vals  = w_rows["tf_share"].to_numpy(dtype="float32")
        w_vec[rel_idx] = w_vals

        sec_win = SECTOR_ARR[start:start+12, :]  # [12, n_sectors]
        pears = np.empty(sec_win.shape[1], dtype="float32")
        coses = np.empty(sec_win.shape[1], dtype="float32")
        for j in range(sec_win.shape[1]):
            s_vec = sec_win[:, j]
            pears[j] = _pearson(w_vec, s_vec)
            coses[j] = _cosine(w_vec, s_vec)

        # rank: primary pearson, tiebreak cosine
        if np.all(np.isnan(pears)) and np.all(np.isnan(coses)):
            out.append({
                "word": w, "first_seen": fs_m, "method": "fixed_after_12m",
                "assigned_sector": None, "pearson": np.nan, "cosine": np.nan,
                "second_sector": None, "second_pearson": np.nan,
                "margin_pearson": np.nan, "ambiguous": True
            })
            continue
        order = np.argsort(np.nan_to_num(pears, nan=-9) + 1e-6*np.nan_to_num(coses, nan=-9))[::-1]
        b = order[0]
        s = order[1] if len(order) > 1 else None
        best_sec = SECTORS[b]
        best_p   = float(pears[b]) if np.isfinite(pears[b]) else float("nan")
        best_c   = float(coses[b]) if np.isfinite(coses[b]) else float("nan")
        second_sec = SECTORS[s] if s is not None else None
        second_p   = float(pears[s]) if (s is not None and np.isfinite(pears[s])) else float("nan")
        margin     = best_p - second_p if (np.isfinite(best_p) and np.isfinite(second_p)) else float("nan")
        ambiguous  = (not np.isfinite(best_p)) or (best_p < 0.2) or (np.isfinite(margin) and margin < 0.05)

        out.append({
            "word": w,
            "first_seen": fs_m,
            "method": "fixed_after_12m",
            "assigned_sector": best_sec,
            "pearson": best_p,
            "cosine": best_c,
            "second_sector": second_sec,
            "second_pearson": second_p,
            "margin_pearson": margin,
            "ambiguous": bool(ambiguous)
        })
    return out

def main():
    import argparse, sys, time, json
    from datetime import datetime
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from multiprocessing import Pool, cpu_count

    # ------- helpers -------
    def log(msg):
        ts = datetime.now().strftime("[%H:%M:%S]")
        print(f"{ts} {msg}")
        sys.stdout.flush()

    def to_month_id(dts: pd.Series) -> pd.Series:
        dts = pd.to_datetime(dts)
        return (dts.dt.year * 12 + (dts.dt.month - 1)).astype("int64")

    # ------- args -------
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--sector_freq", required=True)
    ap.add_argument("--frozen_dict", required=True)
    ap.add_argument("--restrict_to_oos", action="store_true")
    ap.add_argument("--out_dir", default="data/textsec/processed")
    ap.add_argument("--checkpoint_every", type=int, default=50_000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit_words", type=int, default=None)
    ap.add_argument("--n_jobs", type=int, default=max(1, cpu_count() // 2))
    ap.add_argument("--min_points", type=int, default=6, help="min non-zero months in first 12m")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "new_words_sector_mapping_fixed12m.csv"

    # ------- load inputs -------
    log("loading word panel…")
    panel = pd.read_parquet(args.panel, columns=["date","gram","tf_share"])
    log(f"panel rows: {len(panel):,} | unique grams: {panel['gram'].nunique():,}")

    log("loading sector frequency…")
    sf = pd.read_parquet(args.sector_freq, columns=["date","sector","sector_tf_share"])
    sectors = sorted(sf["sector"].unique().tolist())
    log(f"sectors: {len(sectors)} -> {sectors}")

    # ------- month index & sector arrays -------
    log("indexing months…")
    panel["month_id"] = to_month_id(panel["date"])
    sf["month_id"] = to_month_id(sf["date"])
    min_m = min(panel["month_id"].min(), sf["month_id"].min())
    max_m = max(panel["month_id"].max(), sf["month_id"].max())
    n_months = int(max_m - min_m + 1)
    log(f"month_id range: {min_m}..{max_m} ({n_months} months)")

    log("building sector arrays…")
    sector_arr = np.full((n_months, len(sectors)), np.nan, dtype="float32")
    for j, s in enumerate(sectors):
        sj = sf.loc[sf["sector"] == s, ["month_id","sector_tf_share"]]
        idx = (sj["month_id"].to_numpy(dtype="int64") - min_m)
        vals = sj["sector_tf_share"].to_numpy(dtype="float32")
        sector_arr[idx, j] = vals

    # ------- frozen vocab & first_seen -------
    log("loading frozen dict…")
    frozen = json.loads(Path(args.frozen_dict).read_text())
    frozen_words = set(w for vs in frozen.values() for w in vs)

    log("computing first_seen per word…")
    fs = panel.groupby("gram", sort=False)["month_id"].min()   # first_seen month_id
    all_words = set(fs.index.tolist())
    new_words = all_words - frozen_words
    if args.restrict_to_oos:
        cutoff = (pd.Timestamp("2015-01-01").year * 12 + 0)  # Jan 2015
        new_words = {w for w in new_words if fs[w] >= cutoff}
    if args.limit_words:
        new_words = set(sorted(new_words)[:args.limit_words])

    # ------- slice to first 12 months & prune rare -------
    log("slicing panel to first 12 months per word…")
    panel = panel.join(fs.rename("fs_m"), on="gram")
    panel["rel"] = panel["month_id"] - panel["fs_m"]
    panel_12 = panel.loc[(panel["rel"] >= 0) & (panel["rel"] < 12), ["gram","month_id","rel","tf_share"]]

    counts = (panel_12.assign(nz=lambda d: d["tf_share"] > 0)
                        .groupby("gram")["nz"].sum())
    keep_words = set(counts[counts >= args.min_points].index)
    new_words = [w for w in sorted(new_words) if w in keep_words]
    total = len(new_words)
    log(f"after min_points>={args.min_points}: {total:,} words to process")

    # ------- resume handling -------
    processed_words = set()
    wrote_header = False
    if args.resume and out_fp.exists():
        try:
            prev = pd.read_csv(out_fp, usecols=["word"])
            processed_words = set(prev["word"].astype(str).tolist())
            wrote_header = True
            log(f"resume: found {len(processed_words):,} words already mapped")
        except Exception as e:
            log(f"resume: could not read existing {out_fp} ({e}); continuing fresh")

    # ------- batching -------
    BATCH = 10_000
    batches = [new_words[i:i+BATCH] for i in range(0, len(new_words), BATCH)]

    # materialize light state for workers
    fs_idx = {k: int(v) for k, v in fs.items()}

    # ------- writer -------
    processed = 0
    def write_rows(rows):
        nonlocal wrote_header, processed
        if not rows:
            return
        dfb = pd.DataFrame(rows)
        mode = "a" if wrote_header and out_fp.exists() else "w"
        header = not wrote_header
        dfb.to_csv(out_fp, mode=mode, header=header, index=False)
        wrote_header = True
        processed += len(rows)

    # ------- multiprocessing with initializer -------
    log(f"starting mapping with n_jobs={args.n_jobs}, batches={len(batches)}, batch_size≈{BATCH}")
    try:
        with Pool(
            processes=args.n_jobs,
            initializer=init_worker,
            initargs=(panel_12, fs_idx, sector_arr, sectors, min_m, processed_words)
        ) as pool:
            for bi, rows in enumerate(pool.imap_unordered(process_batch, batches, chunksize=1), 1):
                write_rows(rows)
                # checkpoint by words processed (approximate via batches)
                if (processed >= args.checkpoint_every) and (processed % args.checkpoint_every < BATCH):
                    pct = processed / max(1, total)
                    log(f"checkpoint: {processed}/{total} ({pct:.1%}) -> {out_fp}")
    finally:
        pass

    mins = (time.time() - t0) / 60.0
    log(f"done. wrote {out_fp} | processed {processed:,} words | elapsed {mins:.2f} min")


if __name__ == "__main__":
    main()

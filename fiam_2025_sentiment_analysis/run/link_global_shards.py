#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Link GLOBAL shards to (gvkey, iid) using explicit columns from the provided CSV.

CSV columns (explicit):
  - company_col  : e.g., 'conm'
  - country_col  : e.g., 'fic'
  - date_col     : e.g., 'datadate'
  - gvkey_col    : e.g., 'gvkey'
  - iid_col      : e.g., 'iid'

Matching priority (cascading; only fills rows with missing gvkey):
  1) company_norm + country + exact date
  2) company_norm + country + same year
  3) company_norm + country
If 'fic' == 'NAN' or empty, we treat it as missing and allow matches without country on tiers 1–3.

Usage:
  python run/link_global_shards.py \
    --paths config/paths.yaml \
    --name_merge "data/raw/Global_Name_Merge_by_DataDate_GVKEY_IID.csv" \
    --company_col conm --country_col fic --date_col datadate --gvkey_col gvkey --iid_col iid
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
import yaml
from difflib import SequenceMatcher
from typing import List, Optional
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set multiprocessing start method for CUDA-safe spawn
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

STOPWORDS = {
    "INC","INCORPORATED","CORP","CORPORATION","CO","COMPANY","LTD","LIMITED","PLC",
    "AG","NV","SA","S.A.","S.A","OYJ","AB","AS","ASA","KK","K.K.","SE","S.P.A.","SPA",
    "LLC","LP","BHD","HLDGS","GRP"  # <-- removed GROUP and HOLDINGS from here
}
JUNK_TOKENS = {
    "ORDINARY","SHARE","SHARES","CLASS","PREFERENCE","PREFERRED","SERIES","STOCK",
    "REGISTERED","NOM","SPON","ADR","GDR","DR","DEPOSITARY","UNIT","UNITS",
    "ETF","TRUST","HK","USD","EUR","CNY","CNH","A","B","H"
}
SUFFIX_PAT = re.compile(r"[\-\s](A|B|H)$")

# common abbreviations → full tokens (helps HK names)
ABBREV_MAP = {
    "INTL": "INTERNATIONAL",
    "INT'L": "INTERNATIONAL",
    "CO": "COMPANY",
    "COS": "COMPANIES",
    "MGMT": "MANAGEMENT",
    "TECH": "TECHNOLOGY",
    "RES": "RESOURCES",
    "CTR": "CENTER",
    "CTR.": "CENTER",
}

def _expand_abbrevs(s: str) -> str:
    # replace whole-word abbreviations
    for k, v in ABBREV_MAP.items():
        s = re.sub(rf"\b{k}\b", v, s)
    return s

def normalize_name(x: str) -> str:
    """Uppercase, expand abbreviations, drop punctuation, keep GROUP/HOLDINGS, collapse spaces."""
    if not isinstance(x, str):
        x = "" if pd.isna(x) else str(x)
    x = x.upper().strip().replace("&", " AND ")
    x = _expand_abbrevs(x)
    x = SUFFIX_PAT.sub("", x)                          # drop trailing -A/-H etc.
    x = re.sub(r"[^0-9A-Z\s]", " ", x)                 # keep ASCII letters/digits/space
    parts = [p for p in x.split() if p not in STOPWORDS and p not in JUNK_TOKENS]
    x = " ".join(parts)
    return re.sub(r"\s+", " ", x).strip()

def token_set(s: str) -> str:
    toks = s.split()
    return " ".join(sorted(set(toks)))

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, token_set(a), token_set(b)).ratio()

COUNTRY_EQUIV = {
    "HKG": {"HKG","CHN"},
    "CHN": {"CHN","HKG"},
}
NORTH_AMERICA = {"USA","CAN","MEX"}

def equivalent_countries(cty: str) -> set[str]:
    if cty is None or pd.isna(cty):
        return set()
    c = str(cty).upper()
    return COUNTRY_EQUIV.get(c, {c})

def load_merge_table(csv_paths: List[Path],
                     company_col: str,
                     country_col: Optional[str],
                     date_col: Optional[str],
                     gvkey_col: str,
                     iid_col: Optional[str]) -> pd.DataFrame:
    """
    Load and union one or more name-merge CSVs.
    - If a CSV has no country_col (e.g., North America file), nm_country will be NA.
      During linking, nm_country==NA is allowed to match rows whose base country is in NORTH_AMERICA.
    """
    frames = []
    for p in csv_paths:
        nm = pd.read_csv(p, dtype=str, low_memory=False)
        cols = set(nm.columns)

        # Required in every CSV
        need = [company_col, gvkey_col]
        miss = [c for c in need if c not in cols]
        if miss:
            raise SystemExit(f"❌ CSV {p} missing columns: {miss}. Found: {sorted(cols)}")

        nm = nm.copy()
        nm["nm_company_norm"] = nm[company_col].map(normalize_name)
        nm["nm_gvkey"] = nm[gvkey_col].astype(str)
        nm["nm_iid"]   = nm[iid_col].astype(str) if iid_col and iid_col in cols else ""

        # country inference: if present, normalize; else leave NA (handled in linker)
        if country_col and country_col in cols:
            nm["nm_country"] = nm[country_col].astype(str).str.strip().str.upper().replace({"NAN": None, "": None})
        else:
            nm["nm_country"] = pd.NA  # will be allowed for NORTH_AMERICA rows

        # dates (optional)
        if date_col and date_col in cols:
            nm["nm_date"] = pd.to_datetime(nm[date_col], errors="coerce", utc=False)
            nm["nm_year"] = nm["nm_date"].dt.year
        else:
            nm["nm_date"] = pd.NaT
            nm["nm_year"] = pd.NA

        frames.append(nm[["nm_company_norm","nm_country","nm_date","nm_year","nm_gvkey","nm_iid"]])

    if not frames:
        raise SystemExit("No merge CSVs provided.")
    nm_all = pd.concat(frames, ignore_index=True)

    # Deduplicate: prefer most recent date within (company,country,date)
    nm_all = (nm_all
              .sort_values(["nm_company_norm","nm_country","nm_date"])
              .drop_duplicates(subset=["nm_company_norm","nm_country","nm_date"], keep="last"))

    return nm_all.reset_index(drop=True)

def link_one_shard(shard_path: Path, nm: pd.DataFrame) -> dict:
    """Link a shard file, with atomic writes to prevent corruption."""
    import tempfile
    
    # Try to read the file, handle corruption gracefully
    try:
        df = pd.read_parquet(shard_path)
    except Exception as e:
        return {"path": str(shard_path), "rows": 0, "linked": 0, "note": f"read_error: {str(e)[:100]}"}
    
    if df.empty:
        return {"path": str(shard_path), "rows": 0, "linked": 0, "note": "empty"}

    # Build keys on shard
    df["company_name"] = df.get("company_name", "").astype(str)
    df["company_norm"] = df["company_name"].map(normalize_name)
    df["country"]      = df["country"].astype(str).str.strip().str.upper().replace({"NAN": None, "": None})
    df["filing_date"]  = pd.to_datetime(df["filing_date"], errors="coerce", utc=False)
    df["year"]         = df["filing_date"].dt.year

    needs_mask = df["gvkey"].isna()
    if not needs_mask.any():
        return {"path": str(shard_path), "rows": len(df), "linked": 0, "note": "no missing gvkey"}

    # Only rows missing gvkey; keep their original index for alignment
    base = df.loc[needs_mask, ["company_norm","country","filing_date","year"]].copy()
    bidx = base.index

    # ---------- Exact tiers (allowing country crosswalk & NA-without-fic) ----------
    # M1: company + exact date; filter by allowed countries
    right1 = nm[["nm_company_norm","nm_country","nm_date","nm_gvkey","nm_iid"]]
    m1_raw = base.merge(right1,
                        left_on=["company_norm","filing_date"],
                        right_on=["nm_company_norm","nm_date"],
                        how="left").assign(_row_index=lambda x: x.index)
    def _allow(row):
        row_cty = row["country"]
        nm_cty  = row["nm_country"]
        # allow NA CSV (nm_country is NA) only for NORTH_AMERICA
        if pd.isna(nm_cty):
            return (isinstance(row_cty, str) and row_cty in NORTH_AMERICA)
        return nm_cty in equivalent_countries(row_cty)
    m1 = m1_raw[m1_raw.apply(_allow, axis=1)][["_row_index","nm_gvkey","nm_iid"]].set_index("_row_index").reindex(bidx)
    g1 = pd.Series(m1["nm_gvkey"].values, index=bidx)
    i1 = pd.Series(m1["nm_iid"].values,   index=bidx)

    # M2: company + year; same filtering
    right2 = nm[["nm_company_norm","nm_country","nm_year","nm_gvkey","nm_iid"]]
    m2_raw = base.merge(right2,
                        left_on=["company_norm","year"],
                        right_on=["nm_company_norm","nm_year"],
                        how="left").assign(_row_index=lambda x: x.index)
    m2 = m2_raw[m2_raw.apply(_allow, axis=1)][["_row_index","nm_gvkey","nm_iid"]].set_index("_row_index").reindex(bidx)
    g2 = pd.Series(m2["nm_gvkey"].values, index=bidx)
    i2 = pd.Series(m2["nm_iid"].values,   index=bidx)

    # M3: company only; same filtering
    right3 = nm[["nm_company_norm","nm_country","nm_gvkey","nm_iid"]].drop_duplicates(["nm_company_norm","nm_country"])
    m3_raw = base.merge(right3,
                        left_on=["company_norm"],
                        right_on=["nm_company_norm"],
                        how="left").assign(_row_index=lambda x: x.index)
    m3 = m3_raw[m3_raw.apply(_allow, axis=1)][["_row_index","nm_gvkey","nm_iid"]].set_index("_row_index").reindex(bidx)
    g3 = pd.Series(m3["nm_gvkey"].values, index=bidx)
    i3 = pd.Series(m3["nm_iid"].values,   index=bidx)

    gvkey_fill = g1.combine_first(g2).combine_first(g3)
    iid_fill   = i1.combine_first(i2).combine_first(i3)

    # ---------- Constrained fuzzy (country+year buckets) for hard markets ----------
    still = gvkey_fill.isna()
    if still.any():
        tmp = base.loc[still].copy()

        # Build candidate pools by (equivalent country set, year)
        # precompute nm groups
        nm_groups = {}
        for (cty, yr), sub in nm.groupby(["nm_country","nm_year"], dropna=False):
            if pd.isna(yr): 
                continue
            key = (None if pd.isna(cty) else str(cty), int(yr))
            nm_groups.setdefault(key, []).append(
                sub[["nm_company_norm","nm_gvkey","nm_iid"]]
                  .dropna(subset=["nm_company_norm"]).drop_duplicates("nm_company_norm")
            )

        THR = {"CAN": 0.90, "HKG": 0.86, "CHN": 0.88}
        MAX_CANDIDATES = 50000  # Skip fuzzy matching if pool exceeds this (likely too slow)
        MAX_FUZZY_ROWS = 20000  # Limit number of rows to process with fuzzy matching (performance)
        
        # For very large shards, only process a sample for fuzzy matching
        if len(tmp) > MAX_FUZZY_ROWS:
            print(f"    ⚠️  Too many rows for fuzzy matching ({len(tmp)}). Processing first {MAX_FUZZY_ROWS} rows only.", flush=True)
            tmp = tmp.head(MAX_FUZZY_ROWS)
        
        processed_count = 0
        total_to_process = len(tmp)
        
        # More frequent progress for large shards
        progress_interval = 500 if total_to_process > 20000 else 1000
        
        for ridx, row in tmp.iterrows():
            processed_count += 1
            if processed_count % progress_interval == 0:
                pct = (processed_count / total_to_process) * 100
                print(f"    Fuzzy matching progress: {processed_count}/{total_to_process} rows ({pct:.1f}%)...", flush=True)
            
            cty = row["country"]; yr = row["year"]
            if pd.isna(yr):  # need a year scope for safety
                continue
            yr = int(yr)
            # build candidate frames from all equivalent countries; also NA if nm_country is NA
            cand_frames = []
            for c in equivalent_countries(cty):
                k = (c, yr)
                if k in nm_groups:
                    cand_frames.extend(nm_groups[k])
            # include nm_country == NA for NA countries
            if isinstance(cty, str) and cty in NORTH_AMERICA:
                k = (None, yr)
                if k in nm_groups:
                    cand_frames.extend(nm_groups[k])
            if not cand_frames:
                continue

            cand_tbl = (pd.concat(cand_frames, ignore_index=True)
                          .drop_duplicates("nm_company_norm"))
            
            # Skip fuzzy matching if candidate pool is too large (performance safeguard)
            if len(cand_tbl) > MAX_CANDIDATES:
                continue
            
            # pruning + threshold (HK gets a wider window and slightly looser threshold)
            nm0 = row["company_norm"]
            if not nm0:
                continue

            n0 = len(nm0)
            first0 = nm0[:1]

            # Prepare quick features on candidates once
            if "n_len" not in cand_tbl.columns or "first" not in cand_tbl.columns:
                cand_tbl = cand_tbl.assign(
                    n_len=cand_tbl["nm_company_norm"].str.len(),
                    first=cand_tbl["nm_company_norm"].str[:1],
                )

            if cty == "HKG":
                window = 12
                thr = 0.86  # slightly looser for HK; country+year still constrain it
                # HK often has abbreviations/truncations, so skip first-letter pruning
                # But apply stricter size limits for performance (much smaller for HK)
                pool = cand_tbl[cand_tbl["n_len"].between(n0 - window, n0 + window)]
                if pool.empty:
                    pool = cand_tbl
                # For HKG, use much smaller pool limit due to large number of rows
                if len(pool) > 2000:
                    pool = pool.sample(n=2000, random_state=42)
            else:
                window = 8
                thr = {"CAN": 0.90, "CHN": 0.88}.get(cty, 0.92)
                # Use first-letter + length window; then relax first-letter if needed
                pool = cand_tbl[(cand_tbl["first"] == first0) & (cand_tbl["n_len"].between(n0 - window, n0 + window))]
                if pool.empty:
                    pool = cand_tbl[cand_tbl["n_len"].between(n0 - window, n0 + window)]
                    if pool.empty:
                        pool = cand_tbl

            # Limit pool size for all countries (safety) - reduced for performance
            if len(pool) > 5000:
                pool = pool.sample(n=5000, random_state=42)

            best, bg, bi = 0.0, None, None
            for _, cand in pool.iterrows():
                sc = sim(nm0, cand["nm_company_norm"])
                if sc > best:
                    best, bg, bi = sc, cand["nm_gvkey"], cand["nm_iid"]
                # Early exit if we find a perfect match
                if best >= 0.99:
                    break

            if best >= thr:
                gvkey_fill.loc[ridx] = bg
                iid_fill.loc[ridx]   = bi

    # ---------- Write back (StringDtype-safe) ----------
    gvkey_fill_s = gvkey_fill.astype("string")
    iid_fill_s   = iid_fill.astype("string")

    df["gvkey"] = df["gvkey"].astype("string")
    if "iid" not in df.columns:
        df["iid"] = pd.NA
    df["iid"] = df["iid"].astype("string")

    before = df["gvkey"].notna().sum()
    df.loc[needs_mask, "gvkey"] = df.loc[needs_mask, "gvkey"].combine_first(gvkey_fill_s)
    df.loc[needs_mask, "iid"]   = df.loc[needs_mask, "iid"].combine_first(iid_fill_s)
    after  = df["gvkey"].notna().sum()

    df.drop(columns=["company_norm","year"], inplace=True, errors="ignore")
    
    # Atomic write: write to temp file first, then rename (prevents corruption on crash)
    temp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    try:
        df.to_parquet(temp_path, index=False)
        # Atomic rename (Unix/Linux guarantees atomic rename)
        temp_path.replace(shard_path)
    except Exception as e:
        # Clean up temp file if write failed
        if temp_path.exists():
            temp_path.unlink()
        raise

    return {"path": str(shard_path), "rows": len(df), "linked": int(after-before), "note": ""}

# Worker initialization
def _worker_init():
    """Initialize worker process - catch import errors early."""
    try:
        import pandas as pd
        from pathlib import Path
    except ImportError as e:
        import sys
        print(f"  ❌ Import error in worker: {e}", file=sys.stderr, flush=True)
        raise

# Worker wrapper for multiprocessing
def _link_one_shard_worker(args_tuple):
    """Worker wrapper for multiprocessing - unpacks arguments."""
    import traceback
    import sys
    shard_path_str, nm = args_tuple
    try:
        return link_one_shard(Path(shard_path_str), nm)
    except Exception as e:
        print(f"  ⚠️  Error processing shard {Path(shard_path_str).name}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return {"path": shard_path_str, "rows": 0, "linked": 0, "note": f"error: {str(e)}"}

def main():
    ap = argparse.ArgumentParser(description="Link global shards to (gvkey,iid) using one or more name-merge CSVs.")
    ap.add_argument("--paths", default="config/paths.yaml", help="Path to paths.yaml")
    ap.add_argument("--name_merge", required=True, help="Primary name-merge CSV (e.g., Global...)")
    ap.add_argument("--extra_name_merge", action="append", default=[],
                    help="Additional name-merge CSV(s) to union (e.g., North America...). Can be passed multiple times.")
    ap.add_argument("--company_col", required=True, help="Company name column in CSV(s), e.g. 'conm'")
    ap.add_argument("--country_col", default="", help="Country/ISO column in CSV(s), e.g. 'fic' (optional for NA file)")
    ap.add_argument("--date_col", default="", help="Date column in CSV(s), e.g. 'datadate'")
    ap.add_argument("--gvkey_col", required=True, help="GVKEY column in CSV(s), e.g. 'gvkey'")
    ap.add_argument("--iid_col", default="", help="IID column in CSV(s), e.g. 'iid'")
    ap.add_argument("--countries", default="",
                    help="Limit processing to comma-separated ISO3 list (e.g., 'CAN,HKG,CHN'). Default: all.")
    ap.add_argument("--manifest_key", default="filings_clean_manifest",
                    help="Key in paths.yaml under textsec for the shard manifest (default: filings_clean_manifest).")
    ap.add_argument("--num_workers", type=int, default=min(4, multiprocessing.cpu_count()),
                    help="Number of parallel workers (default: min(4, cpu_count) to reduce memory pressure from large merge tables)")
    args = ap.parse_args()

    # Load paths + shard manifest
    paths = yaml.safe_load(open(args.paths))
    manifest_fp = Path(paths["textsec"].get(args.manifest_key, "data/textsec/interim/filings_clean_manifest.csv"))
    if not manifest_fp.exists():
        raise SystemExit(f"❌ Manifest not found: {manifest_fp}. Run make_sec_features.py first.")

    man = pd.read_csv(manifest_fp)
    # Optional country filter
    if args.countries:
        allow = {c.strip().upper() for c in args.countries.split(",") if c.strip()}
        if "country" not in man.columns:
            raise SystemExit("❌ Manifest missing 'country' column for filtering.")
        before = len(man)
        man = man[man["country"].str.upper().isin(allow)]
        print(f"Filtered countries: kept {len(man)}/{before} shards for {sorted(allow)}")

    if man.empty:
        raise SystemExit("❌ No shards to process after filtering.")

    # Load/union name-merge CSVs
    csvs = [Path(args.name_merge)] + [Path(x) for x in args.extra_name_merge]
    nm = load_merge_table(
        csvs,
        company_col=args.company_col,
        country_col=(args.country_col or None),
        date_col=(args.date_col or None),
        gvkey_col=args.gvkey_col,
        iid_col=(args.iid_col or None),
    )

    # Process shards in parallel
    print(f"🚀 Processing {len(man)} shards with {args.num_workers} workers...")
    stats = []
    processed = 0
    shard_paths = man["shard_path"].tolist()
    
    # Prepare arguments: (shard_path_str, nm) tuples
    worker_args = [(sp, nm) for sp in shard_paths]
    
    # Multiprocessing with spawn context for safety
    mp_ctx = multiprocessing.get_context("spawn")
    failed_shards = []
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_ctx, initializer=_worker_init) as ex:
        futs = {ex.submit(_link_one_shard_worker, args): args[0] for args in worker_args}
        for fut in as_completed(futs):
            shard_path_str = futs[fut]
            try:
                res = fut.result(timeout=3600)  # 1 hour timeout per shard
                stats.append(res)
                processed += 1
                if processed % 20 == 0 or processed == len(shard_paths):
                    print(f"  Processed {processed}/{len(shard_paths)} shards…")
            except Exception as e:
                processed += 1
                print(f"  ❌ Failed to get result for {Path(shard_path_str).name}: {type(e).__name__}", flush=True)
                import traceback
                traceback.print_exc()
                failed_shards.append(shard_path_str)
                stats.append({"path": shard_path_str, "rows": 0, "linked": 0, "note": f"failed: {type(e).__name__}"})
    
    # Retry failed shards sequentially
    if failed_shards:
        print(f"\n🔄 Retrying {len(failed_shards)} failed shards sequentially...")
        for shard_path_str in failed_shards:
            shard_path = Path(shard_path_str)
            try:
                print(f"  Retrying {shard_path.name}...")
                
                # Check if file exists and is readable
                if not shard_path.exists():
                    print(f"    ❌ File does not exist: {shard_path}")
                    continue
                
                # Try to read file first to detect corruption
                try:
                    test_df = pd.read_parquet(shard_path)
                except Exception as read_err:
                    print(f"    ❌ File appears corrupted (cannot read): {read_err}")
                    print(f"    💡 Suggestion: Restore this file from backup or regenerate from source")
                    # Keep the failed entry in stats
                    continue
                
                res = link_one_shard(shard_path, nm)
                # Replace the failed entry in stats
                for i, s in enumerate(stats):
                    if s["path"] == shard_path_str and "failed:" in s.get("note", ""):
                        stats[i] = res
                        break
                else:
                    stats.append(res)
                
                if "read_error" in res.get("note", ""):
                    print(f"    ❌ {res['note']}")
                elif res["linked"] > 0:
                    print(f"    ✅ Linked {res['linked']} rows")
                elif res["rows"] > 0:
                    print(f"    ⚠️  Processed {res['rows']} rows (no new links)")
                else:
                    print(f"    ⚠️  {res.get('note', 'no output')}")
            except Exception as e:
                print(f"    ❌ Sequential retry also failed: {e}", flush=True)
                import traceback
                traceback.print_exc()

    # Summary
    out = pd.DataFrame(stats)
    total_new = int(out["linked"].fillna(0).sum())
    print("\n==== LINK SUMMARY ====")
    # Show last 10 rows for a concise glance
    cols = ["path", "rows", "linked", "note"]
    cols = [c for c in cols if c in out.columns]
    print(out[cols].tail(10).to_string(index=False))
    print(f"\n✅ Shards processed: {len(stats)} | Newly linked rows: {total_new}")

if __name__ == "__main__":
    main()

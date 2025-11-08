#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-2 cleaner (streaming, shard-based).

Updates:
- US now writes TWO shard sets per source:
  1) US-fused  : tidy long with single 'text' (rf/mgmt merged)  -> --us_fused_dir
  2) US-split  : preserves separate 'rf' and 'mgmt' (no 'text') -> --us_split_dir
- Non-US writes only fused (single 'text').

Manifest columns: shard_path, country, flavor, source_file

Run global linking AFTER this step (per-shard) to fill (gvkey, iid).
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import yaml
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set multiprocessing start method for CUDA-safe spawn
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

# Map Global file code -> ISO3 (from your 26-country sheet)
FILECODE_TO_ISO3 = {
    "MF":"MEX","XW":"CHE","TT":"TWN","AT":"AUS","AV":"AUT","NO":"NOR","IM":"ITA","GF":"DEU","HK":"HKG",
    "KP":"KOR","JT":"JPN","PL":"PRT","SS":"SWE","SP":"SGP","FP":"FRA","ID":"IRL","DC":"DNK","FH":"FIN",
    "IT":"ISR","SN":"ESP","BB":"BEL","NA":"NLD","CG":"CHN","CT":"CAN","LX":"LUX","LN":"GBR"
}
ISO3_TO_CONTINENT = {
    "USA":"Americas","CAN":"Americas","MEX":"Americas",
    "AUT":"Europe","BEL":"Europe","CHE":"Europe","DEU":"Europe","DNK":"Europe","ESP":"Europe","FIN":"Europe",
    "FRA":"Europe","GBR":"Europe","IRL":"Europe","ITA":"Europe","LUX":"Europe","NLD":"Europe","NOR":"Europe",
    "PRT":"Europe","SWE":"Europe",
    "AUS":"APAC","CHN":"APAC","HKG":"APAC","JPN":"APAC","KOR":"APAC","SGP":"APAC","TWN":"APAC",
    "ISR":"MiddleEast",
}

# US special text columns
US_SPECIAL_TEXT_COLUMNS = {"rf": "risk_factors", "mgmt": "management_discussion_and_analysis"}

# Candidate names
NAME_COLS = ["company_name","name","conm","issuer_name","company"]
DATE_COLS = ["filing_date","date","datadate","period_end_date"]
FORM_COLS = ["form_type","form","report_type","doc_type","type"]

SECTION_PREFIXES = ["section_", "sec_", "mdna_", "mda_", "risk_", "item_", "section ", "item "]
SECTION_REGEXES  = [
    r"^item[\s_]*\d+[a-z]?$", r"^risk[_\s]?factors$", r"^(md&a|mda|mdna)$",
    r"^management[_\s]?discussion", r"^liquidity[_\s]?and[_\s]?capital", r"^business$",
]

def _find_first(cands: List[str], cols: List[str]) -> Optional[str]:
    for c in cands:
        if c in cols: return c
    return None

def _is_section_like(col: str) -> bool:
    c = col.lower().strip()
    if c in {"section","sec","section_name","text"}: return False
    if any(c.startswith(p) for p in SECTION_PREFIXES): return True
    for pat in SECTION_REGEXES:
        if re.match(pat, c): return True
    return False

def _read_any(fp: Path, debug: bool=False) -> Optional[pd.DataFrame]:
    try:
        suf = fp.suffix.lower()
        if suf in (".parquet",".pq"):
            if debug: print(f"  • Read parquet: {fp}")
            return pd.read_parquet(fp)
        if suf in (".csv",):
            if debug: print(f"  • Read csv: {fp}")
            return pd.read_csv(fp)
        if suf in (".jsonl",".ndjson"):
            if debug: print(f"  • Read jsonl: {fp}")
            return pd.read_json(fp, lines=True)
        if suf in (".json",):
            if debug: print(f"  • Read json: {fp}")
            return pd.read_json(fp)
        if suf in (".pkl",".pickle"):
            try:
                if debug: print(f"  • Read pkl: {fp}")
                return pd.read_pickle(fp)
            except Exception as e:
                print(f"  ⚠️  PKL read failed: {fp.name} :: {e}")
                return None
        return None
    except Exception as e:
        print(f"  ⚠️  Read error: {fp.name} :: {e}")
        return None

def _gather_files(root: Path) -> List[Path]:
    if not root.exists(): return []
    return [p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in (".pkl",".pickle",".parquet",".pq",".csv",".json",".jsonl",".ndjson")]

def _force_geo_from_source(df: pd.DataFrame, src: Path) -> pd.DataFrame:
    s = str(src).replace("\\","/")
    if "/Global/" in s or s.endswith("/Global") or "/Global" in s:
        code = src.stem.upper()
        iso  = FILECODE_TO_ISO3.get(code)
        country = iso if iso else None
    elif "/US/" in s or s.endswith("/US") or "/US" in s:
        country = "USA"
    else:
        country = None

    df = df.copy()
    if country is not None:
        df["country"] = country
    df["country"] = df.get("country", pd.Series([None]*len(df))).fillna("USA")
    df["excntry"] = df["country"]
    df["continent"] = df["country"].map(ISO3_TO_CONTINENT).fillna("Other")
    return df

# ---------- FUSED (tidy) normalizer (keeps single 'text') ----------

def _melt_sections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a wide/varied SEC-like dataframe into long format with columns:
      - text   : the section/body text (with US rf/mgmt fallback connector)
      - section: normalized section name (e.g., 'risk_factors', 'management_discussion', etc.)
    Other id/meta columns are preserved.
    """
    cols = list(df.columns)

    # A) US special first (e.g., rf/mgmt columns)
    us_present = [c for c in US_SPECIAL_TEXT_COLUMNS if c in cols]
    if us_present:
        # Build fused rf + "\n\n" + mgmt per original row to use as fallback for empty text
        df = df.copy()
        df["__orig_idx"] = df.index

        rf_series   = df.get("rf",   "")
        mgmt_series = df.get("mgmt", "")
        rf_series   = rf_series.astype(str)
        mgmt_series = mgmt_series.astype(str)

        fused = []
        for a, b in zip(rf_series, mgmt_series):
            parts = []
            if isinstance(a, str) and a.strip(): parts.append(a)
            if isinstance(b, str) and b.strip(): parts.append(b)
            fused.append("\n\n".join(parts) if parts else "")
        df["__fused_text"] = pd.Series(fused, index=df.index)

        id_cols = [c for c in cols if c not in us_present] + ["__orig_idx"]
        m = df.melt(
            id_vars=id_cols,
            value_vars=us_present,
            var_name="__section_key",
            value_name="text",
        )
        m["section"] = m["__section_key"].map(US_SPECIAL_TEXT_COLUMNS).fillna(m["__section_key"])
        m.drop(columns=["__section_key"], inplace=True)

        # Fill empty text cells from the fused rf+mgmt fallback
        m["text"] = m["text"].astype(str)
        mask_empty = m["text"].str.strip().str.len().fillna(0).eq(0)
        if mask_empty.any():
            fused_map = df.set_index("__orig_idx")["__fused_text"]
            m.loc[mask_empty, "text"] = m.loc[mask_empty, "__orig_idx"].map(fused_map).fillna("")

        # Whitespace tidy and cleanup
        m["text"] = m["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        m.drop(columns=["__orig_idx"], inplace=True)
        return m

    # B) Generic "wide" section layout (multiple section-like columns)
    section_like = [c for c in cols if _is_section_like(c)]
    if section_like:
        id_cols = [c for c in cols if c not in section_like]
        m = df.melt(id_vars=id_cols, value_vars=section_like, var_name="section", value_name="text")
        m["text"] = m["text"].astype(str)
        m["section"] = m["section"].astype(str)
        m["text"] = m["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        return m

    # C) Single text-like column present
    for t in ["text", "content", "body", "mdna", "mda", "md&a", "section_text", "rf", "mgmt"]:
        if t in cols:
            d = df.copy()
            d["text"] = d[t].astype(str)
            if "section" not in d.columns:
                d["section"] = US_SPECIAL_TEXT_COLUMNS.get(t, "full_document")
            d["text"] = d["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            return d

    # D) Fallback: best-effort concat of object columns (excluding obvious ids/meta)
    meta_like = {"gvkey", "iid", "company_name", "filing_date", "form_type", "section"}
    longish = [c for c in cols if df[c].dtype == "object" and c not in meta_like]
    d = df.copy()
    d["text"] = d[longish].astype(str).agg("\n\n".join, axis=1) if longish else ""
    if "section" not in d.columns:
        d["section"] = "full_document"
    d["text"] = d["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return d

def _normalize_one_fused(df: pd.DataFrame, src: Path, debug: bool=False) -> pd.DataFrame:
    cols = list(df.columns)
    name_col = _find_first(NAME_COLS, cols)
    date_col = _find_first(DATE_COLS, cols)
    form_col = _find_first(FORM_COLS, cols)

    out = df.copy()
    out["company_name"] = out[name_col].astype(str) if name_col else ""
    out["filing_date"]  = pd.to_datetime(out[date_col], errors="coerce", utc=False) if date_col else pd.NaT
    out["form_type"]    = out[form_col].astype(str) if form_col else ""

    out = _melt_sections(out)
    out["section"] = out["section"].astype(str).fillna("full_document")
    out = _force_geo_from_source(out, src)

    for opt in ["gvkey","iid","cik","ticker","isin","figi"]:
        if opt not in out.columns: out[opt] = pd.NA

    keep = [
        "company_name","filing_date","form_type","section","text",
        "excntry","country","continent","gvkey","iid","cik","ticker","isin","figi"
    ]
    for k in keep:
        if k not in out.columns: out[k] = pd.NA
    out = out[keep].copy()

    if debug:
        print("  ↳ fused mapped:", {"name": name_col, "date": date_col, "form": form_col}, "| rows:", len(out))
    return out

# ---------- US-SPLIT normalizer (preserves rf & mgmt, NO 'text') ----------

def _normalize_us_split(df: pd.DataFrame, src: Path, debug: bool=False) -> Optional[pd.DataFrame]:
    cols = list(df.columns)
    if not {"rf","mgmt"}.issubset(set(c.lower() for c in cols)):
        # Try to align case-insensitive
        colmap = {c.lower(): c for c in cols}
        if "rf" in colmap and "mgmt" in colmap:
            rf_col = colmap["rf"]; mgmt_col = colmap["mgmt"]
        else:
            # No proper US split columns; skip split output
            return None
    else:
        # exact case exists
        rf_col = "rf"; mgmt_col = "mgmt"

    name_col = _find_first(NAME_COLS, cols)
    date_col = _find_first(DATE_COLS, cols)
    form_col = _find_first(FORM_COLS, cols)

    out = df.copy()
    out["company_name"] = out[name_col].astype(str) if name_col else ""
    out["filing_date"]  = pd.to_datetime(out[date_col], errors="coerce", utc=False) if date_col else pd.NaT
    out["form_type"]    = out[form_col].astype(str) if form_col else ""

    # Keep raw rf/mgmt as strings
    out["rf"]   = out[rf_col].astype(str)
    out["mgmt"] = out[mgmt_col].astype(str)

    # No fused 'text' in split variant; ensure it's absent if present
    if "text" in out.columns:
        out.drop(columns=["text"], inplace=True, errors="ignore")

    out = _force_geo_from_source(out, src)

    for opt in ["gvkey","iid","cik","ticker","isin","figi","section"]:
        if opt not in out.columns: out[opt] = pd.NA

    keep = [
        "company_name","filing_date","form_type",
        "rf","mgmt",
        "excntry","country","continent",
        "gvkey","iid","cik","ticker","isin","figi","section"
    ]
    for k in keep:
        if k not in out.columns: out[k] = pd.NA
    out = out[keep].copy()

    if debug:
        print("  ↳ us-split mapped:", {"name": name_col, "date": date_col, "form": form_col}, "| rows:", len(out))
    return out

# ---------- Writers ----------

def _write_parquet(df: pd.DataFrame, out_root: Path, country: str, src: Path) -> Path:
    shard_dir = out_root / country
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_fp = shard_dir / (src.stem + ".parquet")
    df.to_parquet(shard_fp, index=False)
    return shard_fp

def process_and_write_us_dual(src: Path, fused_root: Path, split_root: Path, debug: bool=False) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (fused_fp, split_fp)."""
    raw = _read_any(src, debug=debug)
    if raw is None or raw.empty: return (None, None)

    fused = _normalize_one_fused(raw, src, debug=debug)
    fused = fused if fused is not None else pd.DataFrame()
    fused["country"] = fused.get("country", pd.Series(["USA"]*len(fused)))

    # Split variant (rf/mgmt preserved, no text)
    split = _normalize_us_split(raw, src, debug=debug)

    fused_fp = _write_parquet(fused, fused_root, "USA", src) if not fused.empty else None
    split_fp = _write_parquet(split, split_root, "USA", src) if (split is not None and not split.empty) else None
    return (fused_fp, split_fp)

def process_and_write_non_us(src: Path, fused_root: Path, debug: bool=False) -> Optional[Path]:
    raw = _read_any(src, debug=debug)
    if raw is None or raw.empty: return None
    norm = _normalize_one_fused(raw, src, debug=debug)
    if norm is None or norm.empty: return None
    country = str(norm["country"].iloc[0]) if len(norm) else "UNK"
    return _write_parquet(norm, fused_root, country, src)

# ---------- Multiprocessing Workers ----------

def _worker_init():
    """Initialize worker process - catch import errors early."""
    try:
        import pandas as pd
        import yaml
        from pathlib import Path
    except ImportError as e:
        import sys
        print(f"  ❌ Import error in worker: {e}", file=sys.stderr, flush=True)
        raise

def _process_us_file_worker(args_tuple):
    """Worker wrapper for US files - unpacks arguments."""
    import traceback
    import sys
    import os
    import pandas as pd
    from pathlib import Path
    
    fp_str, fused_root_str, split_root_str, debug = args_tuple
    try:
        # Log start to see if we even get here
        if debug:
            print(f"  [Worker PID {os.getpid()}] Starting {fp_str}", file=sys.stderr, flush=True)
        
        src_path = Path(fp_str)
        fused_root = Path(fused_root_str)
        split_root = Path(split_root_str)
        
        # Check file exists and is readable
        if not src_path.exists():
            print(f"  ⚠️  File does not exist: {fp_str}", file=sys.stderr, flush=True)
            return (None, None)
        
        # Try to read file with memory-aware approach
        if debug:
            print(f"  [Worker PID {os.getpid()}] Reading {src_path.name}...", file=sys.stderr, flush=True)
        
        raw = _read_any(src_path, debug=debug)
        if raw is None or raw.empty:
            if debug:
                print(f"  [Worker PID {os.getpid()}] File {src_path.name} is empty or unreadable", file=sys.stderr, flush=True)
            return (None, None)
        
        if debug:
            print(f"  [Worker PID {os.getpid()}] Loaded {len(raw)} rows from {src_path.name}", file=sys.stderr, flush=True)
        
        # Process fused
        if debug:
            print(f"  [Worker PID {os.getpid()}] Processing fused for {src_path.name}...", file=sys.stderr, flush=True)
        fused = _normalize_one_fused(raw, src_path, debug=debug)
        fused = fused if fused is not None else pd.DataFrame()
        fused["country"] = fused.get("country", pd.Series(["USA"]*len(fused)))
        
        # Process split
        if debug:
            print(f"  [Worker PID {os.getpid()}] Processing split for {src_path.name}...", file=sys.stderr, flush=True)
        split = _normalize_us_split(raw, src_path, debug=debug)
        
        # Write outputs
        if debug:
            print(f"  [Worker PID {os.getpid()}] Writing outputs for {src_path.name}...", file=sys.stderr, flush=True)
        fused_fp = _write_parquet(fused, fused_root, "USA", src_path) if not fused.empty else None
        split_fp = _write_parquet(split, split_root, "USA", src_path) if (split is not None and not split.empty) else None
        
        if debug:
            print(f"  [Worker PID {os.getpid()}] Completed {src_path.name}", file=sys.stderr, flush=True)
        
        return (fused_fp, split_fp)
        
    except MemoryError as e:
        print(f"  ❌ Memory error processing {fp_str}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return (None, None)
    except Exception as e:
        # Always print errors with traceback, even without debug flag
        print(f"  ⚠️  Error processing US file {fp_str}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return (None, None)
    except BaseException as e:
        # Catch everything including KeyboardInterrupt, SystemExit, etc.
        print(f"  ❌ Fatal error in worker processing {fp_str}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise

def _process_global_file_worker(args_tuple):
    """Worker wrapper for global (non-US) files - unpacks arguments."""
    import traceback
    import sys
    fp_str, fused_root_str, debug = args_tuple
    try:
        return process_and_write_non_us(Path(fp_str), Path(fused_root_str), debug=debug)
    except Exception as e:
        # Always print errors with traceback, even without debug flag
        print(f"  ⚠️  Error processing global file {fp_str}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return None
    except BaseException as e:
        # Catch everything including KeyboardInterrupt, SystemExit, etc.
        print(f"  ❌ Fatal error in worker processing {fp_str}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="config/paths.yaml")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--us_split_dir", default="data/textsec/interim/filings_clean_shards_us_split",
                help="Where to write US shards that KEEP rf & mgmt columns (no fused text).")
    ap.add_argument("--us_fused_dir", default="data/textsec/interim/filings_clean_shards",
                help="Where to write shards with a single fused 'text' (all countries + US fused).")
    ap.add_argument("--num_workers", type=int, default=min(8, multiprocessing.cpu_count()),
                    help="Number of parallel workers for global files (default: min(8, cpu_count))")
    ap.add_argument("--us_num_workers", type=int, default=min(2, multiprocessing.cpu_count()),
                    help="Number of parallel workers for US files (default: min(2, cpu_count) to reduce memory pressure)")
    args = ap.parse_args()

    paths = yaml.safe_load(open(args.paths))

    raw_dir = Path(paths["textsec"]["raw_dir"])
    raw_dir_us = Path(paths["textsec"].get("raw_dir_us", raw_dir / "US"))
    raw_dir_global = Path(paths["textsec"].get("raw_dir_global", raw_dir / "Global"))
    fused_dir = Path(paths["textsec"].get("filings_clean_shards", args.us_fused_dir))
    manifest_fp = Path(paths["textsec"].get("filings_clean_manifest", "data/textsec/interim/filings_clean_manifest.csv"))

    # Ensure dirs
    fused_dir.mkdir(parents=True, exist_ok=True)
    Path(args.us_split_dir).mkdir(parents=True, exist_ok=True)
    manifest_fp.parent.mkdir(parents=True, exist_ok=True)

    print("=== SCAN ===")
    print("US root     :", raw_dir_us)
    print("Global root :", raw_dir_global)

    us_files = _gather_files(raw_dir_us)
    gl_files = _gather_files(raw_dir_global)
    print(f"Found {len(us_files)} files under {raw_dir_us}")
    print(f"Found {len(gl_files)} files under {raw_dir_global}")

    written_rows = []
    
    # Prepare arguments for workers
    fused_dir_str = str(fused_dir)
    split_dir_str = str(args.us_split_dir)

    # --- US: dual outputs ---
    print(f"🚀 Processing {len(us_files)} US files with {args.us_num_workers} workers...")
    us_processed = 0
    failed_files = []
    
    if us_files:
        us_worker_args = [(str(fp), fused_dir_str, split_dir_str, args.debug) for fp in us_files]
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.us_num_workers, mp_context=mp_ctx, initializer=_worker_init) as ex:
            futs = {ex.submit(_process_us_file_worker, args): args[0] for args in us_worker_args}
            for fut in as_completed(futs):
                fp_str = futs[fut]
                try:
                    fused_fp, split_fp = fut.result(timeout=3600)  # 1 hour timeout per file
                    us_processed += 1
                    if args.debug or us_processed % 5 == 0:
                        print(f"  Processed US: {us_processed}/{len(us_files)}...")
                    if fused_fp:
                        written_rows.append({"shard_path": str(fused_fp), "country": "USA", "flavor": "fused", "source_file": fp_str})
                    if split_fp:
                        written_rows.append({"shard_path": str(split_fp), "country": "USA", "flavor": "split", "source_file": fp_str})
                except Exception as e:
                    us_processed += 1
                    print(f"  ❌ Failed to get result for {Path(fp_str).name}: {type(e).__name__}", flush=True)
                    failed_files.append(fp_str)
        
        # Retry failed files sequentially
        if failed_files:
            print(f"\n🔄 Retrying {len(failed_files)} failed US files sequentially...")
            for fp_str in failed_files:
                try:
                    print(f"  Retrying {Path(fp_str).name}...")
                    fused_fp, split_fp = process_and_write_us_dual(
                        Path(fp_str), 
                        Path(fused_dir_str), 
                        Path(split_dir_str), 
                        debug=args.debug
                    )
                    if fused_fp:
                        written_rows.append({"shard_path": str(fused_fp), "country": "USA", "flavor": "fused", "source_file": fp_str})
                        print(f"    ✅ Fused output written")
                    if split_fp:
                        written_rows.append({"shard_path": str(split_fp), "country": "USA", "flavor": "split", "source_file": fp_str})
                        print(f"    ✅ Split output written")
                    if not fused_fp and not split_fp:
                        print(f"    ⚠️  No output generated")
                except Exception as e:
                    print(f"    ❌ Sequential retry also failed: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    # --- Global (non-US): fused only ---
    print(f"🚀 Processing {len(gl_files)} global files with {args.num_workers} workers...")
    gl_processed = 0
    if gl_files:
        gl_worker_args = [(str(fp), fused_dir_str, args.debug) for fp in gl_files]
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_ctx, initializer=_worker_init) as ex:
            futs = {ex.submit(_process_global_file_worker, args): args[0] for args in gl_worker_args}
            for fut in as_completed(futs):
                fp_str = futs[fut]
                try:
                    out_fp = fut.result()
                    gl_processed += 1
                    if args.debug or gl_processed % 20 == 0:
                        print(f"  Processed Global: {gl_processed}/{len(gl_files)}...")
                    if out_fp:
                        ctry = Path(out_fp).parent.name
                        written_rows.append({"shard_path": str(out_fp), "country": ctry, "flavor": "fused", "source_file": fp_str})
                except Exception as e:
                    gl_processed += 1
                    print(f"  ❌ Failed to get result for {fp_str}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    if written_rows:
        man = pd.DataFrame(written_rows)
        man.to_csv(manifest_fp, index=False)
        print("✅ Wrote shards:", len(written_rows))
        print("✅ Manifest   :", manifest_fp)
        print("Shard countries (counts):")
        print(man["country"].value_counts().to_string())
        print("Flavors (counts):")
        print(man["flavor"].value_counts().to_string())
    else:
        print("⚠️ No shards written (no readable inputs).")

if __name__ == "__main__":
    main()

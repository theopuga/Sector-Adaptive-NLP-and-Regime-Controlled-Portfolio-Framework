# src/textsec/loader.py
"""
SEC Filings Loader
==================
Purpose:
    Loads yearly `.pkl` files of SEC filings into unified Pandas DataFrames,  
    normalizing fields like `cik`, `filing_date`, and `form_type`.

Functions:
    - load_year_folder(year_dir): Reads all `.pkl` files in a given year directory.
    - load_all_pkls(root): Recursively loads all yearly folders.

Outputs:
    Returns a concatenated DataFrame with standardized columns:
        ['cik', 'filing_date', 'form_type', 'text', 'source_path']

Features:
    - Handles variable data formats (dicts, DataFrames, lists of dicts).
    - Converts all filing dates to UTC-naive timestamps.
    - Ensures CIK normalization (10-digit string format).
"""

from __future__ import annotations
from pathlib import Path
import pickle
import pandas as pd
from typing import Iterable, Dict, Any

def _normalize_record(rec: Dict[str, Any], source_path: str) -> Dict[str, Any]:
    # Flexible key normalization
    cik = rec.get("cik") or rec.get("CIK")
    fdate = rec.get("filing_date") or rec.get("date") or rec.get("filingDate")
    form = rec.get("form_type") or rec.get("form") or rec.get("formType")
    # text fields in various extracts (prefer MD&A/RF if present)
    text = rec.get("text") or rec.get("mda") or rec.get("rf") or rec.get("mgmt")
    return {
        "cik": None if cik is None else str(cik).strip(),
        "filing_date": fdate,
        "form_type": None if form is None else str(form).strip(),
        "text": text,
        "source_path": source_path,
    }

def load_year_folder(year_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(year_dir.glob("*.pkl")):
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if hasattr(obj, "iterrows"):
            for _, r in obj.iterrows():
                rows.append(_normalize_record(r, str(p)))
        elif isinstance(obj, list):
            for r in obj:
                rows.append(_normalize_record(r, str(p)))
        else:
            # single dict or unknown
            if isinstance(obj, dict):
                rows.append(_normalize_record(obj, str(p)))
    if not rows:
        return pd.DataFrame(columns=["cik","filing_date","form_type","text","source_path"])
    df = pd.DataFrame(rows)
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce", utc=True).dt.tz_localize(None)
    return df.dropna(subset=["cik", "filing_date", "text"])

def load_all_pkls(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    dfs = []
    for year_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        dfs.append(load_year_folder(year_dir))
    if not dfs:
        return pd.DataFrame(columns=["cik","filing_date","form_type","text","source_path"])
    out = pd.concat(dfs, axis=0, ignore_index=True)
    # CIK normalize to 10 chars (SEC standard)
    out["cik"] = out["cik"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)
    # Clean up form_type
    out["form_type"] = out["form_type"].fillna("").str.upper()
    return out

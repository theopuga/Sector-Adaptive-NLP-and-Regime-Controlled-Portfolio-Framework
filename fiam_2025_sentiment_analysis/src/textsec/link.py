# src/textsec/link.py
"""
CIK–GVKEY Linking Utility
=========================
Purpose:
    Maps SEC company identifiers (CIK) to Compustat identifiers (GVKEY)  
    for integration with financial datasets.

Inputs:
    - monthly : DataFrame with `cik` column.
    - link_us : CSV link table (US coverage).
    - link_na : Optional CSV link table (North America coverage).

Outputs:
    - DataFrame with `gvkey` column added (and invalid rows dropped).

Notes:
    - Standardizes CIK to 10 digits.
    - Cleans and deduplicates link tables, keeping latest mappings.
    - Merges both US and North American link sources if available.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

def map_cik_to_gvkey(monthly: pd.DataFrame, link_us: str, link_na: str | None = None) -> pd.DataFrame:
    """
    Robust join: try US-only link first, then augment with North America table if provided.
    """
    out = monthly.copy()
    out["cik"] = out["cik"].astype(str).str.zfill(10)

    def _prep(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, dtype={"cik": str, "gvkey": str}, low_memory=False)
        # normalize and drop dupes conservatively
        if "datadate" in df.columns:
            df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
            df = df.sort_values(["cik","gvkey","datadate"]).drop_duplicates(subset=["cik","gvkey"], keep="last")
        df["cik"] = df["cik"].astype(str).str.replace(r"\D","", regex=True).str.zfill(10)
        df["gvkey"] = df["gvkey"].astype(str)
        return df[["cik","gvkey"]].drop_duplicates()

    link = _prep(link_us)
    if link_na and Path(link_na).exists():
        link = pd.concat([link, _prep(link_na)], axis=0, ignore_index=True).drop_duplicates()

    merged = out.merge(link, on="cik", how="left")
    return merged.dropna(subset=["gvkey"])

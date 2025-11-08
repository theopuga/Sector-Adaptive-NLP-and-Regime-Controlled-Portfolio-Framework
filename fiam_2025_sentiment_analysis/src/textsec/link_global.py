# src/textsec/link_global.py
from __future__ import annotations
import pandas as pd
import numpy as np
import re

# Canonical Appendix-A mapping → continent
EXCNTRY_TO_COUNTRY = {
    # Americas
    "USA":"USA","CAN":"CAN","MEX":"MEX",
    # Europe
    "AUT":"AUT","BEL":"BEL","DNK":"DNK","FIN":"FIN","FRA":"FRA","DEU":"DEU","IRL":"IRL",
    "ITA":"ITA","LUX":"LUX","NLD":"NLD","NOR":"NOR","PRT":"PRT","ESP":"ESP","SWE":"SWE","CHE":"CHE","GBR":"GBR",
    # APAC
    "AUS":"AUS","NZL":"NZL","CHN":"CHN","HKG":"HKG","JPN":"JPN","KOR":"KOR","SGP":"SGP","TWN":"TWN",
    # Other (per appendix list in brief)
    "ISL":"ISL"
}
COUNTRY_TO_CONTINENT = {
    **{c:"Americas" for c in ["USA","CAN","MEX"]},
    **{c:"Europe"   for c in ["AUT","BEL","DNK","FIN","FRA","DEU","IRL","ITA","LUX","NLD","NOR","PRT","ESP","SWE","CHE","GBR"]},
    **{c:"APAC"     for c in ["AUS","NZL","CHN","HKG","JPN","KOR","SGP","TWN"]},
    "ISL": "MiddleEast",  # or "Europe" if your brief classifies it differently
}

_punct_re = re.compile(r"[^\w\s]+")

def _norm_name(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    s = _punct_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def ensure_geo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has excntry→country→continent (Appendix-A)."""
    # tolerate multiple source column names
    exc = None
    for cand in ["excntry", "exchange_country", "country_code", "country"]:
        if cand in df.columns:
            exc = cand
            break
    if exc is None:
        # If missing, try to infer: US tree ⇒ "USA"
        df["excntry"] = "USA"
        exc = "excntry"

    df["excntry"] = df[exc].astype(str).str.upper().str[:3]
    df["country"] = df["excntry"].map(EXCNTRY_TO_COUNTRY).fillna(df["excntry"])
    df["continent"] = df["country"].map(COUNTRY_TO_CONTINENT).fillna("Other")
    return df

def map_global_names_to_gvkey(
    df: pd.DataFrame,
    name_merge_csv_path: str,
    name_col_candidates=("company_name","name","conm","company","issuer_name"),
    date_col_candidates=("filing_date","date","datadate","period_end_date"),
) -> pd.DataFrame:
    """
    Fill (gvkey, iid) for NON-US rows using Global (ex CA/US) Name-Merge by DataDate–GVKEY–IID.csv.
    Join key: (excntry, month_bucket, normalized_name).
    Leaves US rows unchanged; never overwrites existing gvkey/iid.
    """
    if df.empty:
        return df

    # geo standardization
    df = ensure_geo_columns(df.copy())

    # locate name & date
    name_col = next((c for c in name_col_candidates if c in df.columns), None)
    date_col = next((c for c in date_col_candidates if c in df.columns), None)
    if name_col is None:
        raise ValueError("No company-name column found in dataframe.")
    if date_col is None:
        raise ValueError("No filing date column found in dataframe.")

    dfx = df.copy()
    dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce", utc=False)
    dfx["month_bucket"] = dfx[date_col].dt.to_period("M").dt.to_timestamp("M")
    dfx["norm_name"] = dfx[name_col].map(_norm_name)

    # Load name-merge
    nm = pd.read_csv(name_merge_csv_path)
    # try to find columns
    nm_name = next((c for c in ["name","company_name","conm"] if c in nm.columns), None)
    nm_date = next((c for c in ["datadate","date","filing_date","period_end_date"] if c in nm.columns), None)
    if nm_name is None or nm_date is None:
        raise ValueError("Name-merge CSV must contain company name and a date column (e.g., 'name' + 'datadate').")

    # Normalize name-merge
    nm["norm_name"] = nm[nm_name].map(_norm_name)
    nm[nm_date] = pd.to_datetime(nm[nm_date], errors="coerce", utc=False)
    nm["month_bucket"] = nm[nm_date].dt.to_period("M").dt.to_timestamp("M")

    # require excntry in name-merge; if missing, fall back to df's excntry on join (safer to require)
    nm_ex = next((c for c in ["excntry","country","exchange_country","country_code"] if c in nm.columns), None)
    if nm_ex is None:
        raise ValueError("Name-merge CSV must include an exchange/country code (e.g., 'excntry').")
    nm["excntry"] = nm[nm_ex].astype(str).str.upper().str[:3]

    # pick ids
    id_g = next((c for c in ["gvkey","GVKEY"] if c in nm.columns), None)
    id_i = next((c for c in ["iid","IID"] if c in nm.columns), None)
    if id_g is None:
        raise ValueError("Name-merge CSV must contain 'gvkey'.")
    if id_i is None:
        # tolerate missing iid, fill later as empty string
        nm["iid"] = ""
        id_i = "iid"

    # Only act on non-US rows with missing gvkey
    mask_target = (dfx["country"] != "USA") & (dfx.get("gvkey").isna() if "gvkey" in dfx.columns else True)
    targ = dfx.loc[mask_target, ["excntry","month_bucket","norm_name"]].copy()
    targ["_row_id"] = targ.index

    # Merge
    merged = targ.merge(
        nm[["excntry","month_bucket","norm_name", id_g, id_i]],
        how="left",
        on=["excntry","month_bucket","norm_name"],
        validate="m:1"
    )
    # Attach back
    fill = merged.set_index("_row_id")[[id_g, id_i]]
    if "gvkey" not in dfx.columns:
        dfx["gvkey"] = np.nan
    if "iid" not in dfx.columns:
        dfx["iid"] = ""

    dfx.loc[fill.index, "gvkey"] = dfx.loc[fill.index, "gvkey"].fillna(fill[id_g])
    dfx.loc[fill.index, "iid"]   = np.where(dfx.loc[fill.index, "iid"].astype(str).str.len()>0,
                                            dfx.loc[fill.index, "iid"],
                                            fill[id_i].fillna(""))

    return dfx

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def parse_dates(series: pd.Series, fmt: str | None):
    s = series
    if pd.api.types.is_numeric_dtype(s):
        s = s.astype("Int64").astype("string")
    if fmt:
        return pd.to_datetime(s, format=fmt, errors="coerce")
    return pd.to_datetime(s, errors="coerce")

def to_month_end(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("M").dt.to_timestamp("M")

def _find_col(candidates, cols):
    for c in candidates:
        if c in cols:
            return c
    return None

def main():
    STOCKS_CSV = "/home/francklin99/Desktop/Coding/mcgillfiam2025/assets/ret_sample.csv"
    CLUSTERS_CSV = "./data/etf_soft_probs.csv"

    SECTORS = [
        "Energy", "Materials", "Industrials", "Consumer_Staples",
        "Consumer_Discretionary", "Healthcare", "Financials",
        "Utilities", "Technology", "Style/Size", "Emerging_Markets",
        "Developed_ex-US", "Commodities",
    ]
    
    SECTOR_MAPPING_TICKER = {
        # US sectors
        "XLE": "Energy",
        "XLB": "Materials",
        "XLI": "Industrials",
        "XLP": "Consumer_Staples",
        "XLY": "Consumer_Discretionary",
        "XLV": "Healthcare",
        "XLF": "Financials",
        "XLU": "Utilities",
        "XLK": "Technology",

        # Style / size
        "IWM": "Style/Size",  # Russell 2000 (small-cap)

        # Regions
        "EEM": "Emerging_Markets",
        "EFA": "Developed_ex-US",

        # Commodities
        "DBC": "Commodities",
    }

    SECTOR_TO_TICKERS = {
        "Energy": "XLE",
        "Materials": "XLB",
        "Industrials": "XLI",
        "Consumer_Staples": "XLP",
        "Consumer_Discretionary": "XLY",
        "Healthcare": "XLV",
        "Financials": "XLF",
        "Utilities": "XLU",
        "Technology": "XLK",
        "Style/Size": "IWM",
        "Emerging_Markets": "EEM",
        "Developed_ex-US": "EFA", 
        "Commodities": "DBC",
    }

    CHUNKSIZE = 800_000
    BIG_DATE_COL = "ret_eom"
    BIG_DATE_FMT = "%Y%m%d"
    BIG_ID_COL = "id"
    CLUSTERS_DATE_COL = "date"
    CLUSTERS_ID_COL = "id"
    HARD_COL = "hard_label"
    SOFT_PREFIX = "prob_"
    SOFT_MIN = 0.35

    clusters = pd.read_csv(CLUSTERS_CSV, low_memory=False)
    clusters[CLUSTERS_DATE_COL] = pd.to_datetime(clusters[CLUSTERS_DATE_COL], errors="coerce")
    clusters[CLUSTERS_DATE_COL] = to_month_end(clusters[CLUSTERS_DATE_COL])
    clusters[CLUSTERS_ID_COL] = clusters[CLUSTERS_ID_COL].astype("string")
    clusters[HARD_COL] = clusters[HARD_COL].astype("string").str.strip()
    hard_as_sector = clusters[HARD_COL].map(SECTOR_MAPPING_TICKER)
    if hard_as_sector.isna().any():
        bad_values = clusters.loc[hard_as_sector.isna(), HARD_COL].unique()
        raise ValueError(f"Unmapped hard_label values: {bad_values}")

    clusters["hard_sector"] = hard_as_sector


    for sector in SECTORS:
        out_csv = f"./etf/ret_sample_{sector.replace(' ', '_')}.csv"
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Fresh output
        if out_path.exists():
            out_path.unlink()
            print(f"[INIT] Deleted existing {out_path}")

        hard_mask = clusters["hard_sector"].eq(sector)
        etf_ticker = SECTOR_TO_TICKERS[sector]
        soft_col = f"{SOFT_PREFIX}{etf_ticker}".replace(" ", "_")
        if soft_col not in clusters.columns:
            raise KeyError(f"Missing soft-prob column '{soft_col}' for sector '{sector}'")
        
        clusters[soft_col] = pd.to_numeric(clusters[soft_col], errors="raise")
        soft_mask = clusters[soft_col].ge(SOFT_MIN)
        keep = clusters.loc[hard_mask | soft_mask, [CLUSTERS_DATE_COL, CLUSTERS_ID_COL]].dropna()

        keep = keep.drop_duplicates()

        print(f"[INFO] Sector keys loaded: {len(keep):,} rows, "
            f"{keep[CLUSTERS_ID_COL].nunique():,} unique ids, "
            f"{keep[CLUSTERS_DATE_COL].dt.to_period('M').nunique():,} months. "
            f"Including correlation data from {soft_col}.")

        wrote_header = False

        reader = pd.read_csv(STOCKS_CSV, chunksize=CHUNKSIZE, dtype={BIG_ID_COL: "string"}, low_memory=False)

        for chunk in reader:
            dt = parse_dates(chunk[BIG_DATE_COL], BIG_DATE_FMT)
            chunk["__date_me"] = to_month_end(dt)
            chunk[BIG_ID_COL] = chunk[BIG_ID_COL].astype("string")

            merged = chunk.merge(keep, left_on=["__date_me", BIG_ID_COL], right_on=[CLUSTERS_DATE_COL, CLUSTERS_ID_COL], how="inner", copy=False, suffixes=("_big", "_clus"))

            if merged.empty:
                continue

            if soft_col in merged.columns:
                merged = merged.rename(columns={soft_col: 'correlation'})
            else:
                raise KeyError(f"Missing correlation column '{soft_col}' for sector '{sector}'")

            cols = merged.columns.tolist()

            # Weird things happen when merging, I could fix it but this is easier
            left_id_col = _find_col(
                [f"{BIG_ID_COL}_big", f"{BIG_ID_COL}_x", BIG_ID_COL],
                cols
            )
            if left_id_col is None:
                raise RuntimeError("Could not locate LEFT/big id column after merge.")

            # Create canonical 'id' from the big ID
            merged["id"] = merged[left_id_col].astype("string")

            # Collect all variants of the right (clusters) id to drop
            right_id_variants = [
                f"{CLUSTERS_ID_COL}_clus",
                f"{CLUSTERS_ID_COL}_y",
                CLUSTERS_ID_COL,
            ]
            # Collect all variants of the left id (except the canonical 'id') to drop
            left_id_variants = [
                f"{BIG_ID_COL}_big",
                f"{BIG_ID_COL}_x",
                BIG_ID_COL,
            ]

            # We keep 'id' only; drop other id variants that actually exist
            drop_id_cols = [c for c in set(right_id_variants + left_id_variants)
                            if c in merged.columns and c != "id"]
            merged = merged.drop(columns=drop_id_cols, errors="ignore")

            # Drop date key variants (but never the data date columns like 'ret_eom' unless they’re keys)
            date_drop_variants = [
                "__date_me",
                CLUSTERS_DATE_COL,
                f"{CLUSTERS_DATE_COL}_clus",
                f"{CLUSTERS_DATE_COL}_big",   # sometimes appears
                f"{CLUSTERS_DATE_COL}_x",     # sometimes appears
            ]
            merged = merged.drop(columns=[c for c in date_drop_variants if c in merged.columns], errors="ignore")

            # Ensure 'id' is first column (if present)
            cols = merged.columns.tolist()
            if "id" in cols:
                cols = ["id"] + [c for c in cols if c != "id"]
                merged = merged.loc[:, cols]
            else:
                raise RuntimeError("Could not locate 'id' column after merge.")

            # Append
            merged.to_csv(out_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True

    print(f"[OK] Wrote filtered rows to {out_path}")

if __name__ == "__main__":
    main()

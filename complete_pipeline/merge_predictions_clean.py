import argparse
import glob
import re
from pathlib import Path
import pandas as pd

def derive_sector_name(path):
    """Extract sector name from file name, e.g. pred_sample99_energy.csv -> energy"""
    name = Path(path).stem
    # m = re.search(r"pred_sample_([^_]+(?:_[^_]+)*)", name)
    m = re.search(r"ret_sample_([a-zA-Z_]+?)__", name)
    return m.group(1) if m else name

def main():
    PRED_GLOB = "./etf/pred/pred_e*.csv"
    OUT_CSV = "./all_prediction.csv"

    files = sorted(glob.glob(PRED_GLOB))
    if not files:
        raise SystemExit(f"No files match {PRED_GLOB}")

    frames = []
    for f in files:
        print(f"[INFO] Found {len(files)} files matching {PRED_GLOB}")
        print(f"[INFO] Reading {f}")
        df = pd.read_csv(f)
        if not {"date","id","xgb"}.issubset(df.columns):
            raise ValueError(f"{f} missing required columns")
        
        # Keep base columns
        base_cols = ["id","date","xgb", "stock_ret"]
        
        # Check for uncertainty columns
        uncertainty_cols = ["xgb_p10", "xgb_p50", "xgb_p90", "q_width", "yhat_var_ens", 
                           "ae_recon_err", "resid_pred", "sigma2_tot", "conf_weight", "score"]
        available_uncertainty = [c for c in uncertainty_cols if c in df.columns]
        
        all_cols = base_cols + available_uncertainty
        df = df[all_cols].copy()
        
        # Rename xgb to pred
        df = df.rename(columns={"xgb": "pred"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        sector = derive_sector_name(f)
        df["sector"] = sector
        df["stock_ret"] = df["stock_ret"].astype(float)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    
    # Determine output columns
    base_out_cols = ["id","date","year","month", "stock_ret", "pred", "sector"]
    uncertainty_out_cols = ["xgb_p10", "xgb_p50", "xgb_p90", "q_width", "yhat_var_ens", 
                            "ae_recon_err", "resid_pred", "sigma2_tot", "conf_weight", "score"]
    available_out = [c for c in uncertainty_out_cols if c in merged.columns]
    
    all_out_cols = base_out_cols + available_out
    merged = merged[all_out_cols].sort_values(["date","id"])
    merged.to_csv(OUT_CSV, index=False)
    
    if available_out:
        print(f"[INFO] Uncertainty columns found: {', '.join(available_out)}")
    print(f"[DONE] Merged {len(files)} files -> {OUT_CSV} with {len(merged)} rows")

if __name__ == "__main__":
    main()

import os
import warnings
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import polars as pl

from sklearn.metrics import mean_squared_error, r2_score

from matplotlib import pyplot as plt

from xgboost import XGBRegressor
import time
import yfinance as yf

import gc

np.random.seed(42)
warnings.filterwarnings("ignore", category=UserWarning)

QUANTILES: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
QUANTILE_LOWER_BOUND = 0.00
QUANTILE_UPPER_BOUND = 1.00

INPUT_FILE = None

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

def process_file(file: str):
    """Process a single file with all the model training and prediction logic."""
    import threading
    thread_id = threading.current_thread().name
    gc.collect()
    if not file.startswith("ret_sample_"):
        return

    INPUT_FILE = "./etf/" + file
    FACTORS_FILE = "/home/francklin99/Desktop/Coding/mcgillfiam2025/assets/factor_char_list.csv"
    MKT_FILE = "/home/francklin99/Desktop/Coding/mcgillfiam2025/assets/mkt_ind.csv"
    parts = INPUT_FILE.replace(".csv", "").replace("/", "_").split("_")[1:]
    UTILITY_NAME = f"{'_'.join(parts)}__q{QUANTILE_LOWER_BOUND:.2f}-{QUANTILE_UPPER_BOUND:.2f}".lower()
    print(f"[{thread_id}] Processing: {UTILITY_NAME}")

    data_df = pl.read_csv(INPUT_FILE, try_parse_dates=True)

    
    gc.collect()

    data_df_vol = data_df.sort(['id', 'ret_eom'])
    data_df_vol = data_df_vol.with_columns([
        pl.col("stock_ret").rolling_std(window_size=3, min_samples=1).over("id").shift(1).alias("vol_3m"),
        pl.col("stock_ret").rolling_std(window_size=6, min_samples=4).over("id").shift(1).alias("vol_6m"),
        pl.col("stock_ret").rolling_std(window_size=12, min_samples=10).over("id").shift(1).alias("vol_12m"),
    ])

    data_df_vol = data_df_vol.with_columns([
        pl.col('stock_ret').log1p().alias('log_ret')
    ])

    from datetime import date
    def date_to_int(d):
        return int(d.strftime('%Y%m%d'))
    
    train = data_df_vol.filter(
        (pl.col('ret_eom') >= date_to_int(date(2005, 1, 1))) & 
        (pl.col('ret_eom') < date_to_int(date(2014, 1, 1)))
    )
    val = data_df_vol.filter(
        (pl.col('ret_eom') >= date_to_int(date(2014, 1, 1))) & 
        (pl.col('ret_eom') < date_to_int(date(2015, 1, 1)))
    )
    test = data_df_vol.filter(
        (pl.col('ret_eom') >= date_to_int(date(2015, 1, 1))) & 
        (pl.col('ret_eom') < date_to_int(date(2026, 1, 1)))
    )

    # Quantile filtering: remove extreme outliers from train/val (not test)
    # This improves model stability during training/validation
    q_low = train['stock_ret'].quantile(QUANTILE_LOWER_BOUND)
    q_high = train['stock_ret'].quantile(QUANTILE_UPPER_BOUND)
    train = train.filter(
        (pl.col('stock_ret') >= q_low) & 
        (pl.col('stock_ret') <= q_high)
    )

    q_low_val = val['stock_ret'].quantile(QUANTILE_LOWER_BOUND)
    q_high_val = val['stock_ret'].quantile(QUANTILE_UPPER_BOUND)
    val = val.filter(
        (pl.col('stock_ret') >= q_low_val) & 
        (pl.col('stock_ret') <= q_high_val)
    )
    
    # NOTE: Test data is NOT filtered - we don't know its distribution in advance

    drop_cols = ['id', 'date_big', 'iid', 'char_date', 'ret_eom', 'year', 'month', 'gvkey', 'stock_ret', 'log_ret', 'correlation']
    
    # Only drop columns that exist
    existing_cols = [col for col in drop_cols if col in train.columns]
    X_train = train.drop(existing_cols)
    y_train = train['log_ret'].to_numpy()
    X_val = val.drop(existing_cols)
    y_val = val['log_ret'].to_numpy()
    X_test = test.drop(existing_cols)
    y_test = test['log_ret'].to_numpy()

    def _select_topk_features_with_xgb(
        X_tr: pl.DataFrame, y_tr: np.ndarray,
        X_va: pl.DataFrame, y_va: np.ndarray,
        feature_names: Optional[List[str]] = None,
        k_list: List[int] = [10, 20, 30, 40, 50, 60, 80, 100],
        seed: int = 42
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Select top-k numeric/bool features by XGB importance.
        """
        # 1) Keep only numeric columns with Polars
        # Select numeric columns (excluding strings, dates, etc.)
        numeric_cols = []
        for col in X_tr.columns:
            dtype = X_tr[col].dtype
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]:
                numeric_cols.append(col)
            elif dtype in [pl.Utf8, pl.Object]:  # Skip string columns
                continue
        
        X_tr_numeric = X_tr.select(numeric_cols)
        feature_names = X_tr_numeric.columns if feature_names is None else feature_names
        
        # Convert to numpy arrays for XGBoost
        X_tr_np = X_tr_numeric.to_numpy()
        # Ensure float64 to handle any remaining issues
        X_tr_np = X_tr_np.astype(np.float64)
        
        X_va_np = X_va.select(X_tr_numeric.columns).to_numpy()
        X_va_np = X_va_np.astype(np.float64)

        # 2) Base model to get importances
        # XGBoost with device="cuda" will automatically handle CPU->GPU data transfer
        base = XGBRegressor(
            random_state=seed,
            n_estimators=300,
            max_depth=None,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",  # Use hist with device="cuda" for GPU (XGBoost 2.0+)
            objective="reg:squarederror",
            eval_metric="rmse",
            device="cuda",  # GPU training
        )
        base.fit(X_tr_np, y_tr, eval_set=[(X_va_np, y_va)], verbose=False)

        importances = getattr(base, "feature_importances_", None)
        if importances is None or len(importances) != len(feature_names):
            importances = np.zeros(len(feature_names), dtype=float)

        order = np.argsort(-importances, kind="stable")
        ranked_feats = [feature_names[i] for i in order]

        # 3) Sweep K values
        best_k, best_r2 = None, -np.inf
        for k in sorted(set(int(x) for x in k_list if x > 0)):
            k = min(k, len(ranked_feats))
            sel = ranked_feats[:k]

            model = XGBRegressor(
                random_state=seed,
                n_estimators=400,
                max_depth=None,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1.0,
                reg_alpha=0.0,
                reg_lambda=1.0,
                tree_method="hist",  # Use hist with device="cuda" for GPU (XGBoost 2.0+)
                objective="reg:squarederror",
                eval_metric="rmse",
                device="cuda",  # GPU training
            )
            sel_idxs = [ranked_feats.index(s) for s in sel]
            X_tr_sel = X_tr_np[:, sel_idxs]
            X_va_sel = X_va_np[:, sel_idxs]
            # XGBoost with device="cuda" will automatically handle CPU->GPU data transfer
            model.fit(X_tr_sel, y_tr, eval_set=[(X_va_sel, y_va)], verbose=False)
            pred_va = model.predict(X_va_sel)
            r2 = r2_score(y_va, pred_va)

            if r2 > best_r2:
                best_r2, best_k = r2, k

        selected = ranked_feats[:best_k]
        return selected, {"best_k": float(best_k), "best_r2": float(best_r2)}

    # Feature selection done ONCE using initial train/val split (2005-2014)
    # Selected features are then reused throughout the rolling window loop
    # No forward-looking bias: selection uses only historical data (2005-2014)
    selected_features, meta = _select_topk_features_with_xgb(
        X_train, y_train,
        X_val, y_val,
        list(X_train.columns),
        k_list=[50, 70, 90, 120, 150]
    )

    # Loop
    out_rows = []
    from datetime import timedelta
    from dateutil.relativedelta import relativedelta
    start = date(2005, 1, 1)
    oos_end = date(2026, 1, 1)
    counter = 0
    while((start + relativedelta(years=11 + counter)) <= oos_end):
            # Rolling window: strictly chronological splits (no forward-looking bias)
            cutoff_train_end = start + relativedelta(years=9 + counter)
            cutoff_val_end = start + relativedelta(years=10 + counter)
            cutoff_test_end = start + relativedelta(years=11 + counter)

            train = data_df_vol.filter(
                (pl.col('ret_eom') >= date_to_int(start)) & 
                (pl.col('ret_eom') < date_to_int(cutoff_train_end))
            )
            val = data_df_vol.filter(
                (pl.col('ret_eom') >= date_to_int(cutoff_train_end)) & 
                (pl.col('ret_eom') < date_to_int(cutoff_val_end))
            )
            test = data_df_vol.filter(
                (pl.col('ret_eom') >= date_to_int(cutoff_val_end)) & 
                (pl.col('ret_eom') < date_to_int(cutoff_test_end))
            )

            # Quantile filtering: use training data quantiles only (no forward-looking bias)
            # This removes extreme outliers from training/validation sets to improve model stability
            q_low = train['log_ret'].quantile(QUANTILE_LOWER_BOUND)
            q_high = train['log_ret'].quantile(QUANTILE_UPPER_BOUND)
            train = train.filter(
                (pl.col('log_ret') >= q_low) & 
                (pl.col('log_ret') <= q_high)
            )

            # Use the same quantile bounds from training for validation (correct: no test data leakage)
            val = val.filter(
                (pl.col('log_ret') >= q_low) & 
                (pl.col('log_ret') <= q_high)
            )
            
            # CRITICAL: Test data is NEVER filtered/clipped - we don't know its distribution in advance
            # Filtering test data would be forward-looking bias since we'd need to know test data quantiles

            cols_to_drop = ['date_big', "id", "iid", "char_date", "ret_eom", 'year', 'month', 'gvkey', 'stock_ret', 'log_ret', 'correlation']
            # Only drop columns that exist
            existing_cols_to_drop = [col for col in cols_to_drop if col in train.columns]
            X_train = train.drop(existing_cols_to_drop)
            y_train = np.log1p(train['stock_ret'].to_numpy())
            X_val = val.drop(existing_cols_to_drop)
            y_val = np.log1p(val['stock_ret'].to_numpy())
            X_test = test.drop(existing_cols_to_drop)
            y_test = test['stock_ret'].to_numpy() # We expoential the log_ret to get the stock_ret when we make predictions

            # Select features and convert to numpy for XGBoost
            # First, let's only keep features that actually exist and are numeric
            available_features = [f for f in selected_features if f in X_train.columns]
            # Double-check they're numeric - exclude string types
            numeric_features = []
            for f in available_features:
                dtype = X_train[f].dtype
                if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]:
                    numeric_features.append(f)
                elif dtype in [pl.Utf8, pl.Object]:
                    continue
            
            # Try to convert to numeric, handling any errors
            try:
                X_train_np = X_train.select(numeric_features).to_numpy()
                X_test_np = X_test.select(numeric_features).to_numpy()
                
                # Ensure float64 to avoid any string conversion issues
                X_train_np = X_train_np.astype(np.float64)
                X_test_np = X_test_np.astype(np.float64)
            except Exception as e:
                print(f"Error converting to numpy with features {numeric_features[:5]}...")
                print(f"Available features: {X_train.columns[:10]}")
                print(f"Train shape before select: {X_train.shape}")
                raise

            models_by_quantile = {}
            # XGBoost with device="cuda" will automatically handle CPU->GPU data transfer
            # The warning about device mismatch is informational - GPU will still be used
            for q in QUANTILES:
                xgb_reg = XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=None,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    tree_method="hist",  # Use hist with device="cuda" for GPU (XGBoost 2.0+)
                    objective='reg:quantileerror',
                    quantile_alpha=q,
                    eval_metric='rmse',
                    device="cuda",  # GPU training
                )

                xgb_reg.fit(X_train_np, y_train)
                models_by_quantile[q] = xgb_reg

            preds_test_log = {}
            for q in QUANTILES:
                # XGBoost will handle CPU->GPU transfer automatically
                preds_test_log[f"p{int(q*100):02d}_log"] = models_by_quantile[q].predict(X_test_np)
            preds_test_log = pl.DataFrame(preds_test_log)

            pred_cols_out = {}
            for q in QUANTILES:
                pred_cols_out[f"yhat_p{int(q*100):02d}"] = np.expm1(preds_test_log[f"p{int(q*100):02d}_log"].to_numpy())

            block = test.select(["year","month","ret_eom","id","stock_ret"]).with_columns([
                pl.col("year").cast(pl.Int64),
                pl.col("month").cast(pl.Int64)
            ])
            for k, arr in pred_cols_out.items():
                block = block.with_columns([pl.Series(name=k, values=arr)])
            taus_sorted = sorted([int(q*100) for q in QUANTILES])
            qcols = [f"yhat_p{t:02d}" for t in taus_sorted if f"yhat_p{t:02d}" in block.columns]
            Q = block.select(qcols).to_numpy()
            Q.sort(axis=1)
            # Update block with sorted quantiles
            for i, col in enumerate(qcols):
                block = block.with_columns([pl.Series(name=col, values=Q[:, i])])
            
            out_rows.append(block)

            counter += 1

    pred_out = pl.concat(out_rows)
    pred_out = pred_out.unique(subset=["year","month","id"], keep="first").sort(["year","month","id"])
    pred_out = pred_out.with_columns([pl.col("ret_eom").alias("date")])
    pred_out = pred_out.drop("ret_eom")
    out_csv_name = f"./etf/pred/pred_quantiles_{UTILITY_NAME}.csv"
    print(f"[{thread_id}] {out_csv_name}")
    pred_out.write_csv(out_csv_name)

    import re

    def _find_quantile_columns(df: pl.DataFrame):
        cols = [c for c in df.columns if c.startswith("yhat_p")]
        cols = sorted(cols, key=lambda s: int(re.findall(r"\d+", s)[0]))
        taus = [int(re.findall(r"\d+", c)[0]) / 100.0 for c in cols]
        return taus, cols

    def _pinball(y: np.ndarray, qhat: np.ndarray, tau: float) -> float:
        d = y - qhat
        return float(np.mean(np.maximum(tau*d, (tau-1.0)*d)))

    metrics_rows = []
    y = pred_out["stock_ret"].to_numpy()

    # 1) Point errors using the median (p50) if present
    if "yhat_p50" in pred_out.columns:
        yhat50 = pred_out["yhat_p50"].to_numpy()
        mae = float(np.mean(np.abs(y - yhat50)))
        rmse = float(np.sqrt(np.mean((y - yhat50)**2)))
        ss_res = float(np.sum((y - yhat50)**2))
        ss_tot = float(np.sum((y - y.mean())**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        print(f"OOS (p50) -> MAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.4f}")
        metrics_rows.append({"metric":"point_mae_p50","value":mae})
        metrics_rows.append({"metric":"point_rmse_p50","value":rmse})
        metrics_rows.append({"metric":"point_r2_p50","value":r2})
    else:
        print("yhat_p50 not found; skipping point metrics.")

    # 2) Pinball loss and hit-rate per quantile
    taus, qcols = _find_quantile_columns(pred_out)
    pl_vals = []
    if qcols:
        print("\n=== Pinball loss & Hit-rate per τ ===")
        for tau, col in zip(taus, qcols):
            qhat = pred_out[col].to_numpy()
            pinball_loss = _pinball(y, qhat, tau)
            hit = float(np.mean(y <= qhat))  # should be ~tau if well-calibrated
            print(f"τ={tau:4.2f}  pinball={pinball_loss:.8f}  hit_rate={hit:.3f}")
            pl_vals.append(pinball_loss)
            metrics_rows.append({"metric":f"pinball_tau_{tau:.2f}","value":pinball_loss})
            metrics_rows.append({"metric":f"hit_rate_tau_{tau:.2f}","value":hit})

        # 3) CRPS approximation (needs ≥3 quantiles)
        if len(taus) >= 3:
            crps = 2.0 * np.trapezoid(pl_vals, x=taus)
            print(f"\nCRPS ≈ {crps:.8f}")
            metrics_rows.append({"metric":"crps","value":crps})

    # 4) Central-interval coverage & sharpness (avg width)
    def _coverage_and_width(df, lo, hi):
        lo_c, hi_c = f"yhat_p{int(lo*100):02d}", f"yhat_p{int(hi*100):02d}"
        if lo_c in df.columns and hi_c in df.columns:
            inside = (df["stock_ret"] >= df[lo_c]) & (df["stock_ret"] <= df[hi_c])
            cov = float(inside.mean())
            width = float((df[hi_c] - df[lo_c]).mean())
            print(f"[p{int(lo*100):02d}–p{int(hi*100):02d}]  coverage={cov:.3f}  avg_width={width:.6f}")
            metrics_rows.append({"metric":f"coverage_p{int(lo*100):02d}_{int(hi*100):02d}","value":cov})
            metrics_rows.append({"metric":f"avg_width_p{int(lo*100):02d}_{int(hi*100):02d}","value":width})

    print("\n=== Interval coverage & sharpness ===")
    for alpha in (0.50, 0.60, 0.70, 0.80, 0.90):
        lo = (1.0 - alpha)/2.0
        hi = 1.0 - lo
        _coverage_and_width(pred_out, lo, hi)

    for alpha in (0.5, 0.6, 0.7, 0.8, 0.9):
        lo = (1.0 - alpha)/2.0; hi = 1.0 - lo
        lo_c, hi_c = f"yhat_p{int(lo*100):02d}", f"yhat_p{int(hi*100):02d}"
        if lo_c in pred_out and hi_c in pred_out:
            inside = (pred_out["stock_ret"] >= pred_out[lo_c]) & (pred_out["stock_ret"] <= pred_out[hi_c])
            print(f"Nominal={alpha:.1f}  Empirical={inside.mean():.3f}")

    # 5) Per-year coverage for p10–p90 (regime drift check)
    if {"year","yhat_p10","yhat_p90"}.issubset(pred_out.columns):
        pred_out = pred_out.with_columns([
            ((pl.col("stock_ret") >= pl.col("yhat_p10")) & (pl.col("stock_ret") <= pl.col("yhat_p90"))).alias("inside_10_90")
        ])
        cov_by_year = pred_out.group_by("year").agg(pl.mean("inside_10_90"))
        print("\n=== Per-year Coverage (p10–p90) ===")
        for row in cov_by_year.iter_rows(named=True):
            yr = row["year"]
            cv = row["inside_10_90"]
            print(f"{int(yr)}: {cv:.3f}")
            metrics_rows.append({"metric":f"coverage_p10_90_year_{int(yr)}","value":float(cv)})

    # 6) Quantile crossing rate (should be ~0 due to your sort)
    if qcols:
        Q = pred_out.select(qcols).to_numpy()
        diffs = np.diff(Q, axis=1)
        crossings = np.any(diffs < -1e-12, axis=1)
        crossing_rate = float(crossings.mean())
        print(f"\nQuantile crossing rate: {crossing_rate:.4f}")
        metrics_rows.append({"metric":"crossing_rate","value":crossing_rate})

    # 7) Save metrics next to predictions
    metrics_df = pl.DataFrame(metrics_rows)
    metrics_path = f"./etf/metrics/metrics_quantiles_{UTILITY_NAME}.csv"
    metrics_df.write_csv(metrics_path)
    print(f"\n[{thread_id}] Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    # Collect all files to process
    files_to_process = [f for f in os.listdir("./etf/") if f.startswith("ret_sample_")]
    
    print(f"Processing {len(files_to_process)} files with 2 workers (multithreading)...")
    
    # Process files in parallel with 2 workers using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all tasks
        futures = {executor.submit(process_file, file): file for file in files_to_process}
        
        # Process completed tasks
        for future in as_completed(futures):
            file = futures[future]
            try:
                future.result()  # This will raise any exceptions that occurred
                print(f"✓ Completed processing: {file}")
            except Exception as e:
                print(f"✗ Error processing {file}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\nAll files processed!")
            
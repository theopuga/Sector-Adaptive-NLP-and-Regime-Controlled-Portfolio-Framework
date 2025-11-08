import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def parse_stock_dates(series: pd.Series, fmt: str | None):
    """Parse date column that may be string, int, or float."""
    s = series
    if np.issubdtype(s.dtype, np.number):
        s = pd.Series(s).round(0).astype("Int64").astype("string")
    try:
        if fmt:
            dt = pd.to_datetime(s, format=fmt, errors="coerce")
        else:
            dt = pd.to_datetime(s, errors="coerce", utc=False)
    except Exception:
        dt = pd.to_datetime(s.astype("string"), format=fmt, errors="coerce") if fmt else pd.to_datetime(s.astype("string"), errors="coerce")
    return dt

def to_month_end(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("M").dt.to_timestamp("M")

def softmax_vec(x, tau=1.0):
    denom = max(1e-8, np.std(x))
    z = (x / denom) / tau
    z -= np.max(z)
    e = np.exp(z)
    return e / e.sum()

def zscore_expanding_past(X: np.ndarray) -> np.ndarray:
    """
    Expanding, past-only standardization per column.
    For row t, z_t uses stats from rows [0..t-1] only.
    If fewer than 2 past obs, z_t = 0.
    X should have no NaNs; safe-guards included.
    """
    T, N = X.shape
    Z = np.zeros((T, N), dtype=float)

    # Welford online per column
    n = np.zeros(N, dtype=int)
    mu = np.zeros(N, dtype=float)
    M2 = np.zeros(N, dtype=float)

    for t in range(T):
        x = X[t, :]

        # compute z using stats up to t-1
        valid = n >= 2
        if valid.any():
            sd = np.zeros(N, dtype=float)
            sd[valid] = np.sqrt(M2[valid] / (n[valid] - 1))
            sd[sd == 0.0] = 1.0
            z = np.zeros(N, dtype=float)
            z[valid] = (x[valid] - mu[valid]) / sd[valid]
            # sanitize non-finite
            z[~np.isfinite(z)] = 0.0
            Z[t, :] = z
        else:
            # no columns have 2 past obs yet -> zeros
            Z[t, :] = 0.0

        # update stats with current row
        # (X is assumed finite; if not, mask them out)
        mask = np.isfinite(x)
        if mask.any():
            xm = x[mask]
            nm = n[mask].astype(float) + 1.0
            delta = xm - mu[mask]
            mu_new = mu[mask] + delta / nm
            M2_new = M2[mask] + delta * (xm - mu_new)
            # commit
            n[mask] = n[mask] + 1
            mu[mask] = mu_new
            M2[mask] = M2_new

    return Z

def fetch_etf_monthly(etf_list):
    frames = []
    for t in etf_list:
        try:
            hist = yf.Ticker(t).history(period="max", interval="1mo", auto_adjust=False)
            if hist.empty:
                print(f"[WARN] {t}: no data", file=sys.stderr)
                continue
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            hist.index.name = "date"
            price_col = "Close"
            out = hist.reset_index()[["date", price_col]].rename(columns={price_col: "close"})
            out = out.dropna(subset=["close"])
            out.insert(0, "id", t)
            out["date"] = out["date"].dt.to_period("M").dt.to_timestamp("M")
            frames.append(out)
        except Exception as e:
            print(f"[ERROR] fetching {t}: {e}", file=sys.stderr)

    if not frames:
        raise RuntimeError("No ETF data fetched.")

    etf = pd.concat(frames, ignore_index=True).sort_values(["id", "date"])
    etf["ret"] = np.log(etf["close"] / etf.groupby("id")["close"].shift(1))
    etf = etf.dropna(subset=["ret"])
    E = etf.pivot(index="date", columns="id", values="ret").sort_index()
    return E

def drop_duplicate_etfs(E, start, end, cutoff):
    mask = (E.index >= pd.Timestamp(start)) & (E.index <= pd.Timestamp(end))
    Ec = E.loc[mask].dropna(how="any", axis=1)

    C = Ec.corr()
    keep = []
    dropped = set()

    for col in C.columns:  # pandas preserves original column order
        if col in dropped:
            continue
        keep.append(col)
        too_close = C.index[(C[col].abs() > cutoff) & (C.index != col)]
        dropped.update(too_close.tolist())

    E2 = E.drop(columns=list(dropped), errors="ignore")
    if dropped:
        print(f"[ETF] Dropped near-duplicate ETFs (|corr|>{cutoff}): {sorted(dropped)}")
    if E2.shape[1] == 0:
        raise RuntimeError("[ETF] All ETFs dropped by de-dup; lower --dup_corr_cut or disable --dedup.")
    print(f"[ETF] Kept ETFs: {list(E2.columns)}")
    return E2

def build_stock_matrix_memory(stock_csv, date_col, date_format, id_col, ret_col):
    """Load ONLY [date_col, id_col, ret_col] in memory, normalize to month-end, group by (date,id)."""
    usecols = [date_col, id_col, ret_col]
    df = pd.read_csv(stock_csv, usecols=usecols)

    dt = parse_stock_dates(df[date_col], date_format)

    df = pd.DataFrame({
        "date": to_month_end(dt),
        id_col: df[id_col].astype("string"),
        ret_col: pd.to_numeric(df[ret_col], errors="coerce"),
    })
    g = df.dropna(subset=["date"])
    S_all = g.pivot(index="date", columns=id_col, values=ret_col).sort_index()
    print(f"[STOCK] Built monthly matrix in memory: shape={S_all.shape}, "
            f"first={S_all.index.min().date() if len(S_all) else None}, last={S_all.index.max().date() if len(S_all) else None}")
    return S_all

def compute_soft_probs_from_matrix(
    S_all: pd.DataFrame,   # (date x id) monthly returns
    E: pd.DataFrame,       # (date x ETF) monthly returns
    etf_names,
    out_csv: Path,
    max_window,
    min_window,
    min_history,
    tau,
):
    etf_dates = E.index.unique().sort_values()
    rows_out = []

    for t_end in etf_dates:
        # STRICTLY PAST window (exclude t_end)
        months = S_all.index[S_all.index < t_end]
        if len(months) < min_window:
            continue

        window_months = months[-max_window:]
        window_start, window_stop = window_months[0], window_months[-1]

        S_win = S_all.loc[window_start:window_stop]
        E_win = E.loc[window_start:window_stop]

        # Align time index
        common_idx = S_win.index.intersection(E_win.index)
        if len(common_idx) < min_window:
            continue
        S_win = S_win.loc[common_idx]
        E_win = E_win.loc[common_idx]

        # Minimum history per stock
        counts = S_win.notna().sum(axis=0)
        keep_cols = counts[counts >= min_history].index.tolist()
        if len(keep_cols) == 0:
            continue
        S_win = S_win[keep_cols]

        # Forward-fill in-window gaps; drop columns still with NaNs afterwards (no backfill)
        S_ff = S_win.ffill(axis=0)
        col_ok = S_ff.notna().all(axis=0)
        S_ff = S_ff.loc[:, col_ok]
        if S_ff.shape[1] == 0:
            continue

        # Ensure enough months remain
        T = S_ff.shape[0]
        if T < min_window:
            continue

        # --- PAST-ONLY Z-SCORES (expanding) ---
        A = zscore_expanding_past(S_ff.to_numpy(copy=False))   # T x N
        B = zscore_expanding_past(E_win.to_numpy(copy=False))  # T x K

        K = B.shape[1]
        if K == 0:
            raise RuntimeError("[ETF] No ETFs left; check de-dup / E index.")

        # Correlation over the (aligned) window
        corr_mat = (A.T @ B) / T              # N x K
        ids = S_ff.columns.to_list()

        hard_idx = np.argmax(corr_mat, axis=1)
        for i, sid in enumerate(ids):
            p = softmax_vec(corr_mat[i, :], tau=tau)
            row = {"date": t_end, "id": sid, "hard_cluster": int(hard_idx[i])}
            for k in range(K):
                row[f"prob_{etf_names[k]}"] = float(p[k])
            rows_out.append(row)

    if not rows_out:
        print("[WARN] No rows produced. Check overlap with ETFs and min_window/min_history.", file=sys.stderr)
        return

    out = pd.DataFrame(rows_out).sort_values(["date", "id"])
    out["hard_label"] = out["hard_cluster"].map({i: etf_names[i] for i in range(len(etf_names))})
    prob_cols = [c for c in out.columns if c.startswith("prob_")]
    out = out[["date", "id", "hard_cluster", "hard_label"] + prob_cols]
    out.to_csv(out_csv, mode="a", header=False, index=False)
    print(f"[OK] Wrote {len(out):,} rows to {out_csv}")

def plot_etf_cumulative_returns(E, start_date, end_date, title_suffix, save_path):
    mask = (E.index >= pd.Timestamp(start_date)) & (E.index < pd.Timestamp(end_date))
    E_filtered = E.loc[mask]
    
    if E_filtered.empty:
        print(f"No data available for period {start_date} to {end_date}")
        return
    
    # Calculate cumulative returns (starting from 1.0)
    cumulative_returns = (1 + E_filtered).cumprod()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each ETF
    for etf in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[etf], 
                label=etf, linewidth=2, alpha=0.8)
    
    plt.title(f'ETF Cumulative Returns {title_suffix}\n{start_date} to {end_date}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (×)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    # Add some statistics
    final_returns = cumulative_returns.iloc[-1]
    best_etf = final_returns.idxmax()
    worst_etf = final_returns.idxmin()
    best_return = final_returns.max()
    worst_return = final_returns.min()
    
    stats_text = f'Best: {best_etf} ({best_return:.2f}×)\nWorst: {worst_etf} ({worst_return:.2f}×)'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_path}")


def main():
    STOCKS_CSV = "/home/francklin99/Desktop/Coding/mcgillfiam2025/assets/ret_sample.csv"
    OUT_CSV = "./data/etf_soft_probs.csv"
    DATE_COL = "ret_eom"
    DATE_FORMAT = "%Y%m%d"
    ID_COL = "id"
    RET_COL = "stock_ret"
    MAX_WINDOW = 24
    MIN_WINDOW = 6
    MIN_HISTORY = 6
    TAU = 1.0
    ETFS = [
        "XLK","XLV","XLF","XLE","XLI","XLY","XLP","XLU","XLB",  # US sectors
        "IWM","EFA","EEM",                                     # style/regions
        "DBC",                                                 # commodities
    ]
    DUP_CORR_CUT = 0.95
    TRAIN_START = "2005-01-01"
    TRAIN_END = "2014-12-31"

    E = fetch_etf_monthly(ETFS)

    E = drop_duplicate_etfs(E, start=TRAIN_START, end=TRAIN_END, cutoff=DUP_CORR_CUT)

    print(f"[ETF] Shape={E.shape}, first={E.index.min().date()}, last={E.index.max().date()}")
    print(f"[ETF] Columns kept: {list(E.columns)}")

    print("\n[PLOT] Creating cumulative return plots...")

    plot_etf_cumulative_returns(
        E=E,
        start_date="2005-01-01",
        end_date="2015-01-01",
        title_suffix="(2005-2015)",
        save_path="./plots/etf_plots_2005_2015.png"
    )

    S_all = build_stock_matrix_memory(
        stock_csv=Path(STOCKS_CSV),
        date_col=DATE_COL,
        date_format=DATE_FORMAT,
        id_col=ID_COL,
        ret_col=RET_COL,
    )

    if S_all.empty:
        raise RuntimeError("[STOCK] Built matrix is empty.")

    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header_cols = ["date", "id", "hard_cluster", "hard_label"] + [f"prob_{c}" for c in E.columns]
    pd.DataFrame(columns=header_cols).to_csv(out_path, index=False)
    print(f"[INIT] Created (or truncated) {out_path} with header.")

    compute_soft_probs_from_matrix(
        S_all=S_all,
        E=E,
        etf_names=list(E.columns),
        out_csv=out_path,
        max_window=MAX_WINDOW,
        min_window=MIN_WINDOW,
        min_history=MIN_HISTORY,
        tau=TAU,
    )

if __name__ == "__main__":
    main()

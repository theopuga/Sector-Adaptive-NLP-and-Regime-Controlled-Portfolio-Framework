import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from hmmlearn.hmm import GaussianHMM
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

# Inputs
OPTS_CSV_PATH = "./test/opts_sentiment_monthly.csv"
BREADTH_CSV_PATH = "./data/spx_breadth_monthly.csv"

# Outputs
OUTDIR_CSV = "./data"
HMM_CSV_PATH = os.path.join(OUTDIR_CSV, "hmm_states_probs_rolling.csv")
SSI_ONLY_CSV_PATH = os.path.join(OUTDIR_CSV, "ssi_monthly_rolling.csv")
SSI_COMPONENTS_CSV_PATH = os.path.join(OUTDIR_CSV, "ssi_components_monthly_rolling.csv")
SSI_ENHANCED_CSV_PATH = os.path.join(OUTDIR_CSV, "ssi_enhanced_monthly_rolling.csv")

# Plots
OUTDIR_PLOTS = "./plots"
HMM_PNG = os.path.join(OUTDIR_PLOTS, "hmm_states_rolling.png")
SSI_BANDS_PNG = os.path.join(OUTDIR_PLOTS, "ssi_bands_rolling.png")
SSI_DERIVS_PNG = os.path.join(OUTDIR_PLOTS, "ssi_derivatives_rolling.png")

# Parameters
PARAMS_HMM = dict(n_components=2, random_state=42, n_iter=1000, covariance_type="full")
INITIAL_TRAIN_START_YEAR = 2005
INITIAL_TRAIN_END_YEAR = 2014
ROLLING_START_YEAR = 2015  # First year to predict
TH_BEAR = +0.5
TH_BULL = -0.5

def rolling_percentile(s: pd.Series, end_date: str, lookback_months: int = 24) -> float:
    """Percentile rank of the last value vs prior lookback (causal)."""
    last_idx = s.index.get_indexer([pd.Timestamp(end_date)], method="pad")[0]
    end = s.index[last_idx]
    start = end - pd.DateOffset(months=lookback_months)
    hist = s[(s.index > start) & (s.index <= end)].dropna()
    if len(hist) < 6:
        return np.nan
    rank = (hist <= hist.iloc[-1]).mean() * 100.0
    return rank

def ema(x: pd.Series, span: int = 2) -> pd.Series:
    """Simple exponential moving average (used for smoothing FearScore)."""
    return x.ewm(span=span, adjust=False, min_periods=1).mean()

def load_spx_series() -> pd.DataFrame:
    sp500 = yf.download("^GSPC", start=f"{INITIAL_TRAIN_START_YEAR}-01-01", progress=False)
    if isinstance(sp500.columns, pd.MultiIndex):
        close = sp500[("Close", "^GSPC")].rename("SPX")
    else:
        close = sp500["Close"].rename("SPX")
    close.index = pd.to_datetime(close.index)
    return close.to_frame()

def read_opts_monthly(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
    # Remove any duplicate dates
    df = df.groupby(df.index).last()
    
    cols_lower = {c.lower(): c for c in df.columns}
    
    raw_keys = ["putcall_vol", "putcall_oi", "vegaoi_tilt", "iv_term_slope", "iv_skew"]
    available_raw = [cols_lower[k] for k in raw_keys if k in cols_lower]
    
    if available_raw:
        use = df[available_raw].copy()
        use.columns = [c.lower() for c in use.columns]
        return use
    elif "ssi_options" in cols_lower:
        s = pd.to_numeric(df[cols_lower["ssi_options"]], errors="coerce")
        return pd.DataFrame({"ssi_options": s})
    else:
        raise ValueError("Options CSV missing both raw option columns and 'SSI_options'.")

def read_breadth_monthly(path: str) -> pd.Series:
    tmp = pd.read_csv(path)
    if tmp.shape[1] == 2 and tmp.columns[0].lower().startswith("date"):
        s = pd.Series(tmp.iloc[:, 1].values,
                      index=pd.to_datetime(tmp.iloc[:, 0].values),
                      name=tmp.columns[1])
    elif "date" in {c.lower() for c in tmp.columns}:
        date_col = [c for c in tmp.columns if c.lower() == "date"][0]
        num_cols = [c for c in tmp.columns if c != date_col]
        s = pd.Series(tmp[num_cols[-1]].values,
                      index=pd.to_datetime(tmp[date_col].values),
                      name=num_cols[-1])
    else:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    
    s = s.sort_index()
    # Convert to month-end and remove duplicates
    s.index = pd.to_datetime(s.index.to_period("M").to_timestamp("M"))
    s = s.groupby(s.index).last()  # Handle any duplicates by taking last value
    return s

def prepare_data(prices: pd.DataFrame, instrument: str, ma: int = 20) -> pd.DataFrame:
    df = prices.copy()
    df[f"{instrument}_ma"] = df[instrument].rolling(ma).mean()
    df[f"{instrument}_log_return"] = np.log(df[f"{instrument}_ma"] / df[f"{instrument}_ma"].shift(1))
    df = df.dropna(subset=[f"{instrument}_log_return"])
    df = df.rename(columns={f"{instrument}_log_return": "x"})
    # Keep only what we need
    return df[["x"]].copy()

def _log_gaussian_prob_1d(x, mean, var):
    # x: (T,), mean: scalar, var: scalar
    # stable log N(x | mean, var)
    return -0.5*(np.log(2*np.pi*var) + ((x - mean)**2)/var)

def _emission_log_probs(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Return log p(x_t | state=j) for all t,j.
    X shape: (T, 1) for 1D obs.
    Supports full or diag covars for 1D.
    """
    T = X.shape[0]
    K = model.n_components
    means = model.means_.reshape(K)  # (K,)
    covars = model.covars_
    # Handle full vs diag for 1D
    if model.covariance_type == "full":
        # covars shape (K, 1, 1)
        vars_ = covars.reshape(K)
    elif model.covariance_type in ("diag", "spherical"):
        vars_ = covars.reshape(K)
    else:
        # tied -> covars shape (1, 1, 1) same for all states
        vars_ = np.repeat(covars.reshape(-1)[0], K)
    x = X[:, 0]  # (T,)
    logB = np.zeros((T, K))
    for j in range(K):
        logB[:, j] = _log_gaussian_prob_1d(x, means[j], vars_[j])
    return logB  # (T, K)

def forward_filter_no_lookahead(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Forward algorithm returning filtered p(state_t | x_1..t) for each t.
    X shape: (T, n_features). For 1D we use n_features=1.
    """
    T = X.shape[0]
    K = model.n_components
    logB = _emission_log_probs(model, X)  # (T, K)

    # log-space to avoid underflow
    log_pi = np.log(model.startprob_ + 1e-300)        # (K,)
    log_A  = np.log(model.transmat_   + 1e-300)       # (K, K)

    log_alpha = np.zeros((T, K))
    # t = 0
    log_alpha[0] = log_pi + logB[0]                   # (K,)
    # normalize
    log_alpha[0] -= np.logaddexp.reduce(log_alpha[0])

    # t >= 1
    for t in range(1, T):
        # log sum over i of alpha_{t-1}(i) + log A(i->j)
        prev = log_alpha[t-1][:, None] + log_A        # (K, K)
        log_alpha[t] = np.logaddexp.reduce(prev, axis=0) + logB[t]
        # normalize
        log_alpha[t] -= np.logaddexp.reduce(log_alpha[t])

    # convert to probabilities
    return np.exp(log_alpha)  # (T, K)

def compute_zscore_params(data: pd.Series, end_date: str) -> tuple:
    """Compute z-score parameters using data up to end_date"""
    mask = data.index <= pd.Timestamp(end_date)
    subset = data.loc[mask].dropna()
    if len(subset) == 0:
        return 0.0, 1.0
    mu = subset.mean()
    sigma = subset.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        sigma = 1.0
    return mu, sigma

def apply_zscore(data: pd.Series, mu: float, sigma: float) -> pd.Series:
    """Apply z-score transformation"""
    return (data - mu) / sigma

def classify_ssi(ssi: pd.Series, th_bear=TH_BEAR, th_bull=TH_BULL) -> pd.Series:
    """Classify SSI into Bull/Neutral/Bear"""
    lab = pd.Series(index=ssi.index, dtype="int8")
    lab.loc[ssi >= th_bear] = -1  # Bearish
    lab.loc[ssi <= th_bull] = +1  # Bullish
    lab = lab.fillna(0)  # Neutral
    return lab

def run_rolling_window_analysis():
    """
    Rolling-window HMM (daily) + SSI (monthly) with phase-aware options overlay.
    Returns: hmm_consolidated (daily), ssi_consolidated (monthly), ssi_enhanced (monthly), spx_daily (daily)
    Expects helpers defined elsewhere: ema(), rolling_percentile(), compute_zscore_params(), apply_zscore(),
    load_spx_series(), read_opts_monthly(), read_breadth_monthly(), prepare_data(), forward_filter_no_lookahead(),
    compute_ssi_derivatives_integrals().
    """
    print("Starting rolling window analysis...")

    # ---------- Load data ----------
    spx_daily = load_spx_series()
    opts_monthly = read_opts_monthly(OPTS_CSV_PATH)          # monthly DF, columns lowercased in loader
    breadth_monthly = read_breadth_monthly(BREADTH_CSV_PATH) # monthly Series

    # ---------- Prepare daily HMM data ----------
    hmm_data = prepare_data(spx_daily, "SPX", ma=20)
    hmm_data["year"] = hmm_data.index.year

    max_year = max(
        hmm_data.index.max().year,
        opts_monthly.index.max().year,
        breadth_monthly.index.max().year
    )

    # =========================================================
    # HMM (daily) — expanding, causal (no look-ahead)
    # =========================================================
    print(f"\n--- HMM Analysis ({INITIAL_TRAIN_START_YEAR}-{max_year}) ---")

    train_start_year = INITIAL_TRAIN_START_YEAR  # 2005
    cutoff_year = INITIAL_TRAIN_END_YEAR         # 2014

    mask_past = (hmm_data["year"] >= train_start_year) & (hmm_data["year"] <= cutoff_year)
    X_past_df = hmm_data.loc[mask_past, ["x"]]
    X_past = X_past_df.values
    years_fwd = sorted(hmm_data.loc[hmm_data["year"] >= (cutoff_year + 1), "year"].unique())

    print(f"Training on {train_start_year}-{cutoff_year}: {len(X_past)} observations")
    print(f"Rolling forward for years: {years_fwd}")

    model = GaussianHMM(**PARAMS_HMM)
    model.fit(X_past)

    # Filtered states for the initial period
    probs_past = forward_filter_no_lookahead(model, X_past)
    states_past = probs_past.argmax(axis=1)
    past_df = pd.DataFrame({"date": X_past_df.index, "state": states_past}).set_index("date")
    for k in range(model.n_components):
        past_df[f"p_state_{k}"] = probs_past[:, k]

    # Roll forward year by year
    out_states = []
    for y in years_fwd:
        X_year_df = hmm_data.loc[hmm_data["year"] == y, ["x"]].copy()
        if X_year_df.empty:
            continue
        X_year = X_year_df.values

        print(f"Processing year {y}: {len(X_year)} daily observations")

        probs_year = forward_filter_no_lookahead(model, X_year)
        states_year = probs_year.argmax(axis=1)

        tmp = pd.DataFrame({"date": X_year_df.index, "state": states_year}).set_index("date")
        for k in range(model.n_components):
            tmp[f"p_state_{k}"] = probs_year[:, k]
        out_states.append(tmp)

        # Refit on data up to year-end (expanding, still causal)
        X_to_date = hmm_data.loc[hmm_data.index <= X_year_df.index[-1], ["x"]].values
        refit = GaussianHMM(**{**PARAMS_HMM, "init_params": ""})
        refit.startprob_ = model.startprob_.copy()
        refit.transmat_  = model.transmat_.copy()
        refit.means_     = model.means_.copy()
        refit.covars_    = model.covars_.copy()
        refit.fit(X_to_date)
        model = refit

    hmm_consolidated = pd.concat([past_df] + out_states).sort_index()
    print(f"Total HMM results: {len(hmm_consolidated)} daily observations")
    print(f"State 1 count: {int((hmm_consolidated['state'] == 1).sum())}")

    # =========================================================
    # SSI (monthly) — phase-aware options + breadth, rolling
    # =========================================================
    print(f"\n--- SSI Analysis (Rolling {ROLLING_START_YEAR}-{max_year}) ---")

    rolling_years = list(range(ROLLING_START_YEAR, max_year + 1))
    ssi_results = []

    for year in rolling_years:
        print(f"Processing SSI for year {year}...")

        # Causal training end
        train_end_date = f"{year-1}-12-31"
        predict_start_date = f"{year}-01-01"
        predict_end_date = f"{year}-12-31"

        # Slice monthly windows
        ssi_pred_mask = (opts_monthly.index >= pd.Timestamp(predict_start_date)) & \
                        (opts_monthly.index <= pd.Timestamp(predict_end_date))
        breadth_pred_mask = (breadth_monthly.index >= pd.Timestamp(predict_start_date)) & \
                            (breadth_monthly.index <= pd.Timestamp(predict_end_date))

        opts_pred = opts_monthly.loc[ssi_pred_mask]
        breadth_pred = breadth_monthly.loc[breadth_pred_mask]

        if len(opts_pred) == 0 or len(breadth_pred) == 0:
            print(f"  No SSI prediction data for {year}")
            continue

        # ---------- Options component (two cases) ----------
        opt_cols = [c.lower() for c in opts_pred.columns]

        # Case A: raw components exist (anything besides ssi_options)
        if set(opt_cols) - {"ssi_options"}:
            # Single sign flip BEFORE z-scoring (backwardation ⇒ stress)
            opts_processed = opts_pred.copy()
            if "iv_term_slope" in opts_processed.columns:
                opts_processed["iv_term_slope"] = -opts_processed["iv_term_slope"]

            # Causal z-scores using training window
            opts_components_zscore = pd.DataFrame(index=opts_processed.index)
            for col in opts_processed.columns:
                col_lower = col.lower()
                mu, sigma = compute_zscore_params(opts_monthly[col], train_end_date)
                opts_components_zscore[col_lower] = apply_zscore(opts_processed[col], mu, sigma)
                print(f"  Options {col}: μ={mu:.3f}, σ={sigma:.3f}")

            # Build fear score from whatever components are present
            use_cols = [c for c in ["vix", "vvix", "iv_skew", "iv_term_slope", "putcall_oi", "putcall_vol"]
                        if c in opts_components_zscore.columns]
            Z = opts_components_zscore[use_cols].copy()

            # Level (smoothed) + trend
            FearScore = Z.mean(axis=1).rename("FearScore_raw")
            FearScore_smooth = ema(FearScore, span=2).rename("FearScore_smooth")
            dFear = FearScore_smooth.diff(1).rename("dFear")

            # Causal 24m percentile
            fear_pct = []
            for dt in FearScore_smooth.index:
                pct = rolling_percentile(FearScore_smooth.loc[:dt],
                                         end_date=dt.strftime("%Y-%m-%d"),
                                         lookback_months=24)
                fear_pct.append(pct)
            fear_pct = pd.Series(fear_pct, index=FearScore_smooth.index, name="FearScore_pct")

            # Phase-aware multiplier
            SSI_opt = pd.Series(1.0, index=FearScore.index, name="SSI_opt")
            panic_mask   = (fear_pct > 85) & (dFear > 0)   # fear rising fast
            relief_mask  = (fear_pct > 60) & (dFear <= 0)  # fear high but easing
            complacent   = (fear_pct < 30)
            SSI_opt.loc[panic_mask]  = 0.7
            SSI_opt.loc[relief_mask] = 1.3
            SSI_opt.loc[complacent]  = 0.9

            # Level-only component used downstream
            ssi_options_level = FearScore_smooth.rename("SSI_options_level")

        # Case B: only composite 'ssi_options' exists
        else:
            mu, sigma = compute_zscore_params(opts_monthly["ssi_options"], train_end_date)
            comp_z = apply_zscore(opts_pred["ssi_options"], mu, sigma).rename("ssi_options_z")
            comp_z = comp_z.groupby(comp_z.index).first()  # dedup
            print(f"  Options composite: μ={mu:.3f}, σ={sigma:.3f}")

            # Build analogs so logic stays uniform
            ssi_options_level = ema(comp_z, span=2).rename("SSI_options_level")
            dFear = ssi_options_level.diff(1).rename("dFear")

            fear_pct = []
            for dt in ssi_options_level.index:
                pct = rolling_percentile(ssi_options_level.loc[:dt],
                                         end_date=dt.strftime("%Y-%m-%d"),
                                         lookback_months=24)
                fear_pct.append(pct)
            fear_pct = pd.Series(fear_pct, index=ssi_options_level.index, name="FearScore_pct")

            SSI_opt = pd.Series(1.0, index=ssi_options_level.index, name="SSI_opt")
            panic_mask   = (fear_pct > 85) & (dFear > 0)
            relief_mask  = (fear_pct > 60) & (dFear <= 0)
            complacent   = (fear_pct < 30)
            SSI_opt.loc[panic_mask]  = 0.7
            SSI_opt.loc[relief_mask] = 1.3
            SSI_opt.loc[complacent]  = 0.9

        # ---------- Breadth (inverse, z-scored causally) ----------
        mu_breadth, sigma_breadth = compute_zscore_params(breadth_monthly, train_end_date)
        z_breadth_inv = -apply_zscore(breadth_pred, mu_breadth, sigma_breadth).rename("z_breadth_inv")
        print(f"  Breadth: μ={mu_breadth:.3f}, σ={sigma_breadth:.3f}")

        # Dedup month-ends if any
        ssi_options_level = ssi_options_level.groupby(ssi_options_level.index).first()
        z_breadth_inv     = z_breadth_inv.groupby(z_breadth_inv.index).first()
        SSI_opt           = SSI_opt.groupby(SSI_opt.index).first()

        # ---------- Combine (aligned inner) ----------
        ssi_components = pd.DataFrame({
            "SSI_options_level": ssi_options_level,
            "z_breadth_inv": z_breadth_inv
        }).dropna()

        ssi_components = ssi_components.join(SSI_opt, how="inner")

        # Base (level-only), then apply phase-aware multiplier
        ssi_base = ssi_components[["SSI_options_level", "z_breadth_inv"]].mean(axis=1).rename("SSI_base")
        ssi_components["SSI"] = (ssi_base * ssi_components["SSI_opt"]).rename("SSI")

        ssi_components["year"] = year
        ssi_results.append(ssi_components)

        print(f"  SSI processed {len(ssi_components)} monthly observations for {year}")

    # ---------- Consolidate SSI ----------
    print("\n--- Consolidating Results ---")
    if ssi_results:
        ssi_consolidated = pd.concat(ssi_results).sort_index()
        print("Computing SSI derivatives and integrals...")
        ssi_enhanced = compute_ssi_derivatives_integrals(ssi_consolidated)
        print(f"Total SSI results: {len(ssi_consolidated)} monthly observations")
    else:
        ssi_consolidated = pd.DataFrame()
        ssi_enhanced = pd.DataFrame()
        print("No SSI results to consolidate")

    return hmm_consolidated, ssi_consolidated, ssi_enhanced, spx_daily

def compute_ssi_derivatives_integrals(ssi_df: pd.DataFrame) -> pd.DataFrame:
    """Compute derivatives and integrals for SSI in a causal manner"""
    ssi_enhanced = ssi_df[["SSI"]].copy()
    
    # First derivative (monthly change)
    ssi_enhanced["SSI_d1"] = ssi_enhanced["SSI"].diff(1)
    
    # Second derivative (change in change)
    ssi_enhanced["SSI_d2"] = ssi_enhanced["SSI_d1"].diff(1)
    
    # First integral (cumulative sum)
    ssi_enhanced["SSI_I1"] = ssi_enhanced["SSI"].cumsum()
    
    # Second integral (cumulative sum of cumulative sum)
    ssi_enhanced["SSI_I2"] = ssi_enhanced["SSI_I1"].cumsum()
    
    return ssi_enhanced

def plot_hmm_states(spx_daily: pd.DataFrame, hmm_df: pd.DataFrame, output_path: str):
    """Plot HMM states overlaid on SPX"""
    # Align SPX with HMM data (accounting for 20-day MA lag)
    spx_aligned = spx_daily.iloc[20:].copy()  # Skip first 20 days due to MA
    plot_data = spx_aligned.join(hmm_df[["state"]], how="inner")
    
    if len(plot_data) == 0:
        print("No data to plot for HMM states")
        return
    
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    # Color mapping for states
    colors = {0: "#1f77b4", 1: "#ff7f0e"}
    
    # Shade regions by state
    states = plot_data["state"].astype(int)
    start_idx = 0
    
    for i in range(1, len(states) + 1):
        if i == len(states) or states.iloc[i] != states.iloc[start_idx]:
            ax.axvspan(plot_data.index[start_idx], plot_data.index[i-1], 
                      color=colors[states.iloc[start_idx]], alpha=0.08, linewidth=0)
            start_idx = i
    
    # Plot SPX price
    ax.plot(plot_data.index, plot_data["SPX"], lw=1.6, color="black", label="SPX Close")
    
    ax.set_title("Rolling Window HMM States on S&P 500", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis
    years = pd.date_range(plot_data.index.min(), plot_data.index.max(), freq="YS")
    if len(years) > 0:
        ax.set_xticks(years)
        ax.set_xticklabels([y.year for y in years], rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved HMM plot: {output_path}")

def plot_ssi_bands(ssi_df: pd.DataFrame, output_path: str):
    """Plot SSI with regime bands"""
    if len(ssi_df) == 0:
        print("No SSI data to plot")
        return
        
    labels = classify_ssi(ssi_df["SSI"], TH_BEAR, TH_BULL)
    
    label_names = {+1: "Bullish", 0: "Neutral", -1: "Bearish"}
    label_colors = {+1: "#18a558", 0: "#bfbfbf", -1: "#d9534f"}
    
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    # Shade regions by regime
    state = labels.values
    start_idx = 0
    
    for i in range(1, len(state) + 1):
        if i == len(state) or state[i] != state[start_idx]:
            ax.axvspan(ssi_df.index[start_idx], ssi_df.index[i-1],
                      color=label_colors[state[start_idx]], alpha=0.15, linewidth=0)
            start_idx = i
    
    # Plot SSI line
    ax.plot(ssi_df.index, ssi_df["SSI"], lw=2, color="black", label="SSI (composite)")
    
    # Add threshold lines
    ax.axhline(TH_BEAR, color=label_colors[-1], ls="--", lw=1, 
              label=f"Bearish threshold = {TH_BEAR:+.1f}")
    ax.axhline(TH_BULL, color=label_colors[+1], ls="--", lw=1, 
              label=f"Bullish threshold = {TH_BULL:+.1f}")
    ax.axhline(0.0, color="grey", ls=":", lw=1)
    
    # Create legend
    patches = [Patch(color=label_colors[k], alpha=0.3, label=label_names[k]) 
              for k in (+1, 0, -1)]
    leg1 = ax.legend(handles=patches, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(loc="upper right")
    
    ax.set_title("Rolling Window SSI Sentiment Regimes", fontsize=14)
    ax.set_ylabel("SSI (z-score, bearish-positive)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    years = pd.date_range(ssi_df.index.min(), ssi_df.index.max(), freq="YS")
    if len(years) > 0:
        ax.set_xticks(years)
        ax.set_xticklabels([y.year for y in years], rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved SSI bands plot: {output_path}")

def plot_ssi_derivatives(enhanced_df: pd.DataFrame, output_path: str):
    """Plot SSI derivatives and integrals"""
    if len(enhanced_df) == 0:
        print("No enhanced SSI data to plot")
        return
        
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # First derivative
    axes[0].plot(enhanced_df.index, enhanced_df["SSI_d1"], lw=1.6, color='blue')
    axes[0].axhline(0, color="grey", ls=":", lw=1)
    axes[0].set_title("SSI First Derivative (Monthly Δ)", fontsize=12)
    axes[0].set_ylabel("Δ SSI")
    axes[0].grid(True, alpha=0.3)
    
    # Second derivative
    axes[1].plot(enhanced_df.index, enhanced_df["SSI_d2"], lw=1.6, color='red')
    axes[1].axhline(0, color="grey", ls=":", lw=1)
    axes[1].set_title("SSI Second Derivative (Monthly Δ²)", fontsize=12)
    axes[1].set_ylabel("Δ² SSI")
    axes[1].grid(True, alpha=0.3)
    
    # Integrals
    axes[2].plot(enhanced_df.index, enhanced_df["SSI_I1"], lw=1.6, 
                label="First Integral (Cumulative SSI)", color='green')
    axes[2].plot(enhanced_df.index, enhanced_df["SSI_I2"], lw=1.2, 
                label="Second Integral (Cumulative of Cumulative)", color='purple')
    axes[2].axhline(0, color="grey", ls=":", lw=1)
    axes[2].set_title("SSI Integrals (Cumulative Sentiment Drift)", fontsize=12)
    axes[2].set_ylabel("Level")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("Date")
    
    # Format x-axis for bottom subplot
    years = pd.date_range(enhanced_df.index.min(), enhanced_df.index.max(), freq="YS")
    if len(years) > 0:
        axes[2].set_xticks(years)
        axes[2].set_xticklabels([y.year for y in years], rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved SSI derivatives plot: {output_path}")


def main():
    """Main execution function"""
    os.makedirs(OUTDIR_CSV, exist_ok=True)
    os.makedirs(OUTDIR_PLOTS, exist_ok=True)
    
    print("="*60)
    print("ROLLING WINDOW CAUSAL HMM + SSI PIPELINE")
    print("="*60)
    
    # Run rolling window analysis
    hmm_results, ssi_results, ssi_enhanced, spx_daily = run_rolling_window_analysis()
    
    # Save results
    print("\n--- Saving Results ---")
    
    if not hmm_results.empty:
        hmm_save = hmm_results.copy()
        hmm_save.index.name = "date"
        hmm_save.to_csv(HMM_CSV_PATH)
        print(f"Saved HMM results: {HMM_CSV_PATH}")
        
        # Plot HMM
        plot_hmm_states(spx_daily, hmm_results, HMM_PNG)
    else:
        print("No HMM results to save")
    
    if not ssi_results.empty:
        # Save SSI files
        ssi_results[["SSI"]].to_csv(SSI_ONLY_CSV_PATH)
        ssi_results.to_csv(SSI_COMPONENTS_CSV_PATH)
        print(f"Saved SSI results: {SSI_ONLY_CSV_PATH}")
        print(f"Saved SSI components: {SSI_COMPONENTS_CSV_PATH}")
        
        # Plot SSI bands
        plot_ssi_bands(ssi_results, SSI_BANDS_PNG)
        
        if not ssi_enhanced.empty:
            ssi_enhanced.to_csv(SSI_ENHANCED_CSV_PATH)
            print(f"Saved SSI enhanced: {SSI_ENHANCED_CSV_PATH}")
            
            # Plot derivatives
            plot_ssi_derivatives(ssi_enhanced, SSI_DERIVS_PNG)
    else:
        print("No SSI results to save")
    
    print("\n" + "="*60)
    print("ROLLING WINDOW ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
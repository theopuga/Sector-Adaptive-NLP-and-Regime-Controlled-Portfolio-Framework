# SSI-Based Portfolio

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Import from config
import config
from config import (
    compute_portfolio_metrics,
    compute_capm_metrics,
    compute_turnover_new,
    compute_net_returns,
    compute_trading_costs,
    compute_cap_composition,
    compute_sector_exposures,
    compute_country_exposures,
    compute_pnl_evolution,
    load_benchmark_data as load_benchmark_data_config,
    check_all_constraints,
    MIN_STOCKS,
    MAX_STOCKS,
    TARGET_TURNOVER,
    LEVERAGE_LONG,
    LEVERAGE_SHORT,
    MAX_SECTOR_NET_EXPOSURE,
    MAX_POSITION_SIZE,
    LARGE_CAP_THRESHOLD,
    INITIAL_AUM,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
L = logging.getLogger("rollopt")

SECTOR_ETFS: Dict[str, str] = {
    "XLK": "technology", "XLV": "healthcare", "XLF": "financials", "XLE": "energy",
    "XLI": "industrials", "XLY": "consumer_discretionary", "XLP": "consumer_staples",
    "XLU": "utilities", "XLB": "materials",
    "IWM": "small_caps", "EFA": "developed_ex_us", "EEM": "emerging_markets",
    "DBC": "commodities",
}

RETURNS_CSV = "./all_prediction_1.csv"
BACKTEST_START = 2017
BACKTEST_END = 2026
OPTIONS_DATA_END_DATE = "2026-01-01"
LOOKBACK_YEARS = 10
OUT_FILE = "./rolling_results.csv"
PATH_TO_WORD_MOMENTUM = "/home/francklin99/Desktop/Coding/mcgillfiam2025/final_copy/sector_word_momentum_monthly_progressive.parquet"
SSI_CSV = "./data/ssi_components_monthly_rolling.csv"
HMM_CSV = "./data/hmm_states_probs_rolling.csv"

MIN_ACTIVE_FRAC = 0.10
DECAY_RATE = None # Will be inferred from training data
OPTIONS_CSV = "./etf_options_new.csv"
LAMBDA_FORECAST = 0.90
MKT_FILE = "../assets/mkt_ind.csv"
MSCI_BENCHMARK_FILE = "../assets/MSCI World Index Return.csv"
IS_MAX_ALLOCATION = True

# Momentum configuration
MOMENTUM_ENABLED = True
MOMENTUM_SECTORS = ["energy", "materials", "commodities"] # Slow moving sectors
MOMENTUM_WEIGHT = 0.5 
# SSI allocation
BEARISH_THRESHOLD = 0.40
BULLISH_THRESHOLD = -0.40
SSI_LONG_BEARISH, SSI_SHORT_BEARISH = 0.3, 0.7
SSI_LONG_BULLISH, SSI_SHORT_BULLISH = 0.85, 0.15

SSI_D1 = 0.3
SSI_D2 = 0.5
SSI_I1 = 0.2
SSI_I2 = 0.2

# HMM allocation
HMM_STATE_0 = 0
HMM_STATE_1 = 1
HMM_P_STATE_0 = 0.5
HMM_P_STATE_1 = 0.5
HMM_REGIME_FACTOR = 0.2
HMM_REGIME_FACTOR_MULTIPLIER = 0.8
HMM_REGIME_MULTIPLIER = 1.5

def _month_end(s) -> pd.Timestamp:
    return pd.to_datetime(s) + pd.offsets.MonthEnd(0)

def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fetch_etf_monthly(etfs: List[str]) -> pd.DataFrame:
    L.info("Fetching ETF monthly from Yahoo for %d tickers", len(etfs))
    frames = []
    for t in etfs:
        hist = yf.Ticker(t).history(period="max", interval="1mo", auto_adjust=False)
        if hist.empty or "Close" not in hist.columns:
            raise RuntimeError(f"{t}: no usable data")
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        temp = hist.reset_index()
        dtcol = temp.columns[0]
        out = temp[[dtcol, "Close"]].rename(columns={dtcol: "date", "Close": "close"})
        out["date"] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp("M")
        out.insert(0, "id", t)
        out = out.dropna(subset=["close"])
        frames.append(out)

    etf = pd.concat(frames, ignore_index=True).sort_values(["id", "date"])
    etf["ret"] = np.log(etf["close"] / etf.groupby("id")["close"].shift(1))
    etf = etf.dropna(subset=["ret"])
    pivot = etf.pivot(index="date", columns="id", values="ret").sort_index()
    L.info("ETF monthly returns matrix: %s (%s .. %s)", pivot.shape, pivot.index.min().date(), pivot.index.max().date())
    return pivot

def get_monthly_etf_returns_from_pivot(E: pd.DataFrame) -> pd.DataFrame:
    long = E.reset_index().melt(id_vars=['date'], var_name='ticker', value_name='monthly_return')
    long['sector'] = long['ticker'].map(SECTOR_ETFS)
    long = long.dropna(subset=['sector']).sort_values(['date','ticker']).reset_index(drop=True)
    long['date'] = long['date'] + pd.offsets.MonthEnd(0)
    return long

def load_returns_data(path: str, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df["date"] = _month_end(df[date_col])
    # prefer EV as the model signal
    if "ev_final" in df.columns:
        df["pred"] = pd.to_numeric(df["ev_final"], errors="coerce")
    else:
        df["pred"] = pd.to_numeric(df.get("pred", np.nan), errors="coerce")
    df["stock_ret"] = pd.to_numeric(df["stock_ret"], errors="coerce")
    df["sector"] = df["sector"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["stock_ret", "pred", "sector"])
    return df[["date", "id", "sector", "pred", "stock_ret"]]

def load_ssi(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = _month_end(df["date"])
    return df

def load_hmm_states(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = _month_end(df["date"])
    df["month_year"] = df["date"].dt.to_period("M")
    out = df.groupby("month_year").last().reset_index()
    out["date"] = out["month_year"].dt.to_timestamp() + pd.offsets.MonthEnd(0)
    return out.drop("month_year", axis=1)

def load_word_momentum(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = _month_end(df["date"])
    
    # Normalize sector names to match portfolio sectors
    if "sector" in df.columns:
        df["sector"] = df["sector"].str.lower().str.replace("-", "_")
        # Handle specific mappings
        sector_mapping = {
            "developed_ex_us": "developed_ex_us", 
            "emerging_markets": "emerging_markets",
        }
        df["sector"] = df["sector"].replace(sector_mapping)
    
    L.info("Word momentum loaded: %d rows, date_range=[%s, %s]", 
           len(df), df["date"].min(), df["date"].max())
    return df

def compute_ssi_allocation(
    ssi_value: float, ssi_d1: float, ssi_d2: float,
    ssi_i1: float, ssi_i2: float,
    ssi_opt: float = 1.0,                 # <-- NEW: options-phase multiplier
    bearish_threshold: float = BEARISH_THRESHOLD,
    bullish_threshold: float = BULLISH_THRESHOLD,
    gross_caps: tuple[float, float] = (0.8, 1.2),
    net_cap: float = 0.95
) -> tuple[float, float]:
    # --- Base long/short from your original SSI value/derivatives ---
    if ssi_value >= bearish_threshold:
        base_long, base_short = SSI_LONG_BEARISH, SSI_SHORT_BEARISH
    elif ssi_value <= bullish_threshold:
        base_long, base_short = SSI_LONG_BULLISH, SSI_SHORT_BULLISH
    else:
        ssi_norm = (ssi_value - bullish_threshold) / (bearish_threshold - bullish_threshold)
        accel = np.tanh(ssi_d2) * SSI_D2
        speed = np.tanh(ssi_d1) * SSI_D1
        integ = np.tanh(ssi_i1) * SSI_I1
        short_adj = np.clip(ssi_norm + accel + speed + integ, 0.0, 1.0)
        base_long, base_short = 1.0 - short_adj, short_adj

    total = base_long + base_short
    if total > 0:
        base_long, base_short = base_long / total, base_short / total

    # --- Apply phase-aware adjustment ---
    net  = np.clip(base_long - base_short, -net_cap, net_cap)
    gross = 1.0

    # Net adjustment: multiply by SSI_opt
    net_adj = np.clip(net * ssi_opt, -net_cap, net_cap)

    # Gross adjustment: map SSI_opt [0.7,1.3] → gross [0.8,1.2]
    lo, hi = gross_caps
    ssi_opt_clamped = np.clip(ssi_opt, 0.7, 1.3)
    gross_mult = lo + (hi - lo) * (ssi_opt_clamped - 0.7) / (1.3 - 0.7)
    gross = np.clip(gross * gross_mult, lo, hi)

    # --- Rebuild long/short ---
    long_w  = 0.5 * gross * (1.0 + net_adj)
    short_w = 0.5 * gross * (1.0 - net_adj)

    long_w  = max(0.0, float(long_w))
    short_w = max(0.0, float(short_w))
    return long_w, short_w


def compute_ssi_hmm_allocation(ssi_value: float, ssi_d1: float, ssi_d2: float,
                               ssi_i1: float, ssi_i2: float,
                               hmm_state: int, p_state_0: float, p_state_1: float,
                               bearish_threshold: float = BEARISH_THRESHOLD,
                               bullish_threshold: float = BULLISH_THRESHOLD) -> Tuple[float, float]:
    regime_factor = HMM_REGIME_FACTOR
    if hmm_state == 1:
        bearish_threshold *= HMM_REGIME_FACTOR_MULTIPLIER
        bullish_threshold *= HMM_REGIME_FACTOR_MULTIPLIER
        regime_multiplier = HMM_REGIME_MULTIPLIER
    else:
        regime_multiplier = 1.0

    if ssi_value >= bearish_threshold:
        base_long, base_short = SSI_LONG_BEARISH, SSI_SHORT_BEARISH
    elif ssi_value <= bullish_threshold:
        base_long, base_short = SSI_LONG_BULLISH, SSI_SHORT_BULLISH
    else:
        ssi_norm = (ssi_value - bullish_threshold) / (bearish_threshold - bullish_threshold)
        accel = np.tanh(ssi_d2) * SSI_D2
        speed = np.tanh(ssi_d1) * SSI_D1
        integ = np.tanh(ssi_i1) * SSI_I1
        hmm = (p_state_1 - p_state_0) * regime_factor
        short_adj = np.clip(ssi_norm + accel + speed + integ + hmm, 0.0, 1.0)
        base_long, base_short = 1.0 - short_adj, short_adj

    if hmm_state == 1:
        if base_short > SSI_SHORT_BULLISH:
            base_short = min(0.9, base_short * HMM_REGIME_MULTIPLIER)
            base_long = 1.0 - base_short
        else:
            base_long = min(0.9, base_long * HMM_REGIME_MULTIPLIER)
            base_short = 1.0 - base_long

    total = base_long + base_short
    if total > 0:
        base_long, base_short = base_long / total, base_short / total
    return base_long, base_short


def smooth_allocation(target_long: float, target_short: float,
                     prev_long: Optional[float], prev_short: Optional[float],
                     max_change_per_month: float = 0.15,
                     smoothing_alpha: float = 0.7) -> Tuple[float, float]:

    if prev_long is None or prev_short is None:
        # First month: use target directly
        return target_long, target_short
    
    # Apply exponential smoothing
    smoothed_long = smoothing_alpha * prev_long + (1.0 - smoothing_alpha) * target_long
    smoothed_short = smoothing_alpha * prev_short + (1.0 - smoothing_alpha) * target_short
    
    # Constrain maximum change per month
    max_change_long = abs(target_long - prev_long)
    max_change_short = abs(target_short - prev_short)
    
    if max_change_long > max_change_per_month:
        direction_long = 1.0 if target_long > prev_long else -1.0
        smoothed_long = prev_long + direction_long * max_change_per_month
        smoothed_long = np.clip(smoothed_long, 0.0, 1.0)
    
    if max_change_short > max_change_per_month:
        direction_short = 1.0 if target_short > prev_short else -1.0
        smoothed_short = prev_short + direction_short * max_change_per_month
        smoothed_short = np.clip(smoothed_short, 0.0, 1.0)
    
    # Renormalize to sum to 1.0 (approximately)
    total = smoothed_long + smoothed_short
    if total > 1e-6:
        smoothed_long = smoothed_long / total
        smoothed_short = smoothed_short / total
    
    return smoothed_long, smoothed_short

def compute_sector_bounds(n_sectors: int, min_active_frac: float = 0.20) -> Tuple[float, float]:
    equal = 1.0 / max(1, n_sectors)
    w_max = min(0.40, max(0.15, 2.5 / max(1, n_sectors)))
    w_min = min_active_frac * equal
    if w_min * n_sectors > 1.0 or w_min > w_max:
        w_min = 0.0
    return w_min, w_max

def compute_max_turnover_from_ssi(
    ssi_value: float,
    bearish_threshold: float = BEARISH_THRESHOLD,
    bullish_threshold: float = BULLISH_THRESHOLD,
    min_turnover: float = 0.15,
    max_turnover: float = 0.45,
) -> float:
    """
    Compute maximum turnover based on SSI value.
    High SSI (bearish) -> higher turnover (more aggressive rebalancing)
    Low SSI (bullish) -> lower turnover (more stable portfolio)
    """
    # Normalize SSI to [0, 1] range where 0 = bullish, 1 = bearish
    if ssi_value >= bearish_threshold:
        # Very bearish: use max turnover
        return max_turnover
    elif ssi_value <= bullish_threshold:
        # Very bullish: use min turnover
        return min_turnover
    else:
        # Linear interpolation between thresholds
        ssi_norm = (ssi_value - bullish_threshold) / (bearish_threshold - bullish_threshold)
        # ssi_norm is now in [0, 1] where 0 = bullish, 1 = bearish
        turnover = min_turnover + ssi_norm * (max_turnover - min_turnover)
        return float(np.clip(turnover, min_turnover, max_turnover))

def rebuild_portfolio_with_turnover_constraint(
    g: pd.DataFrame,
    prev_long_ids: set,
    prev_short_ids: set,
    n_long_target: int,
    n_short_target: int,
    max_turnover: float = 0.45,
    score_col: str = "pred_sec",
) -> Tuple[float, List[str], List[str]]:
    """
    Rebuild portfolio with turnover constraint.
    Keeps at least (1 - max_turnover) of current portfolio, removing worst performers first.
    
    For longs: Keep best performers (highest predictions)
    For shorts: Keep worst performers (lowest predictions)
    """
    # Get current portfolio stocks that exist in this month's data
    g_dict = g.set_index("id")
    prev_long_in_data = [sid for sid in prev_long_ids if sid in g_dict.index]
    prev_short_in_data = [sid for sid in prev_short_ids if sid in g_dict.index]
    
    # Calculate how many to keep (at least 55% if max_turnover = 0.45)
    min_keep_frac = 1.0 - max_turnover
    
    # For LONGS: Keep the best performers (highest predictions)
    kept_long_ids = []
    if prev_long_in_data:
        prev_long_df = g_dict.loc[prev_long_in_data].copy()
        prev_long_df = prev_long_df.sort_values(score_col, ascending=False)
        n_keep_long = max(1, int(np.ceil(len(prev_long_in_data) * min_keep_frac)))
        n_keep_long = min(n_keep_long, len(prev_long_in_data), n_long_target)
        kept_long_ids = prev_long_df.head(n_keep_long).index.tolist()
    
    # For SHORTS: Keep the worst performers (lowest predictions)
    kept_short_ids = []
    if prev_short_in_data:
        prev_short_df = g_dict.loc[prev_short_in_data].copy()
        prev_short_df = prev_short_df.sort_values(score_col, ascending=True)  # ascending=True for worst first
        n_keep_short = max(1, int(np.ceil(len(prev_short_in_data) * min_keep_frac)))
        n_keep_short = min(n_keep_short, len(prev_short_in_data), n_short_target)
        kept_short_ids = prev_short_df.head(n_keep_short).index.tolist()
    
    # Calculate how many new stocks we need
    n_new_long = max(0, n_long_target - len(kept_long_ids))
    n_new_short = max(0, n_short_target - len(kept_short_ids))
    
    # Get available stocks (not already in kept positions)
    kept_all = set(kept_long_ids) | set(kept_short_ids)
    available = g[~g["id"].isin(kept_all)].copy()
    
    if available.empty or (n_new_long == 0 and n_new_short == 0):
        # If no new stocks available or no new stocks needed, return what we have
        final_long_ids = kept_long_ids[:n_long_target]
        final_short_ids = kept_short_ids[:n_short_target]
    else:
        # Use sector-neutral allocation for new stocks
        _, new_long_ids, new_short_ids = build_long_short_portfolio_sector_neutral(
            available, n_new_long, n_new_short, sector_weights=None, score_col=score_col
        )
        
        final_long_ids = kept_long_ids + new_long_ids
        final_short_ids = kept_short_ids + new_short_ids
    
    # Trim to exact targets if we exceeded
    final_long_ids = final_long_ids[:n_long_target]
    final_short_ids = final_short_ids[:n_short_target]
    
    # Calculate returns
    if not final_long_ids or not final_short_ids:
        return np.nan, final_long_ids, final_short_ids
    
    long_rets = g_dict.loc[final_long_ids, "stock_ret"].values
    short_rets = g_dict.loc[final_short_ids, "stock_ret"].values
    
    ret = float(np.mean(long_rets) - np.mean(short_rets))
    return ret, final_long_ids, final_short_ids

def build_long_short_portfolio_sector_neutral(
    g: pd.DataFrame,
    n_long_total: int,
    n_short_total: int,
    sector_weights: Optional[Dict[str, float]] = None,
    min_per_sector: int = 1,
    score_col: str = "pred_sec",
) -> Tuple[float, List[str], List[str]]:
    """
    Pick longs/shorts *within each sector* using `score_col`, then combine.
    Returns zero-cost spread = mean(long returns) - mean(short returns).
    """
    g = g.dropna(subset=[score_col, "stock_ret", "sector"])
    if g.empty:
        return np.nan, [], []

    sectors_present = sorted(g["sector"].unique().tolist())
    if not sectors_present:
        return np.nan, [], []

    # If no sector weights provided, allocate heads roughly equally by sector size
    if sector_weights is None:
        # proportional to universe size per sector
        counts = g.groupby("sector").size()
        w = (counts / counts.sum()).to_dict()
    else:
        # use provided weights (clip to >=0 and renormalize)
        raw = {s: max(0.0, float(sector_weights.get(s, 0.0))) for s in sectors_present}
        ssum = sum(raw.values()) or 1.0
        w = {s: raw.get(s, 0.0)/ssum for s in sectors_present}

    # allocate integers per side
    def _alloc(total_names: int) -> Dict[str, int]:
        base = {s: w[s] * total_names for s in sectors_present}
        floor = {s: int(np.floor(base[s])) for s in sectors_present}
        rem = {s: base[s] - floor[s] for s in sectors_present}
        # distribute the remainder
        k = total_names - sum(floor.values())
        order = sorted(sectors_present, key=lambda s: rem[s], reverse=True)
        alloc = floor.copy()
        for s in order[:max(0, k)]:
            alloc[s] += 1
        # ensure minimum
        for s in sectors_present:
            if alloc[s] < min_per_sector and total_names >= len(sectors_present):
                alloc[s] = min_per_sector
        # if we exceeded budget, pare back largest
        while sum(alloc.values()) > total_names:
            donor = max(alloc, key=alloc.get)
            alloc[donor] -= 1
        return alloc

    nL_by_sector = _alloc(n_long_total)
    nS_by_sector = _alloc(n_short_total)

    long_ids, short_ids = [], []
    long_rets, short_rets = [], []

    for s in sectors_present:
        sg = g[g["sector"] == s].sort_values(score_col, ascending=False)
        nL = min(nL_by_sector.get(s, 0), len(sg))
        nS = min(nS_by_sector.get(s, 0), len(sg))
        if nL > 0:
            Ls = sg.head(nL)
            long_ids.extend(Ls["id"].astype(str).tolist())
            long_rets.extend(Ls["stock_ret"].astype(float).tolist())
        if nS > 0:
            Ss = sg.tail(nS)
            short_ids.extend(Ss["id"].astype(str).tolist())
            short_rets.extend(Ss["stock_ret"].astype(float).tolist())

    if not long_rets or not short_rets:
        return np.nan, long_ids, short_ids

    # equal-weight sector-neutral spread
    ret = float(np.mean(long_rets) - np.mean(short_rets))
    return ret, long_ids, short_ids

def compute_exponential_weights(n_periods: int, decay_rate: float) -> np.ndarray:
    w = np.exp(-decay_rate * np.arange(n_periods)[::-1])
    return w / w.sum()

def estimate_decay_rate_from_training(sector_returns: pd.DataFrame,
                                      hl_min_months: int = 6, hl_max_months: int = 60) -> float:
    rhos = []
    for c in sector_returns.columns:
        s = sector_returns[c].dropna()
        if len(s) > 12:
            rhos.append(float(s.autocorr(lag=1)))
    if not rhos:
        return np.log(2)/24.0
    rho_med = float(np.median(np.clip(rhos, 0.0, 0.95)))
    if rho_med <= 0:
        return np.log(2)/12.0
    hl = max(hl_min_months, min(hl_max_months, -1.0/np.log(rho_med)))
    return float(np.log(2)/hl)

def allocate_sector_cardinality(weights: Dict[str, float], total_names: int) -> Dict[str, int]:
    sectors = list(weights.keys())
    raw = {s: weights[s] * total_names for s in sectors}
    n_floor = {s: int(np.floor(raw[s])) for s in sectors}
    rem = {s: raw[s] - n_floor[s] for s in sectors}
    k = total_names - sum(n_floor.values())
    order = sorted(sectors, key=lambda s: (-rem[s], -weights[s]))
    n = n_floor.copy()
    for s in order[:max(0, k)]:
        n[s] += 1
    for s in sectors:
        if weights[s] > 0 and n[s] == 0:
            donor = max(n, key=n.get)
            if n[donor] > 1:
                n[donor] -= 1
                n[s] = 1
    return n

def dynamic_total_names_from_dispersion(preds: pd.Series, base: int = 175,
                                        min_n: int = 100, max_n: int = 250) -> int:
    p = preds.dropna().values
    if p.size == 0:
        return base
    z = (p - p.mean()) / (p.std() + 1e-12)
    disp = np.std(z)
    x = np.clip((disp - 0.5) / 1.0, 0.0, 1.0)
    N = int(round(max_n - x * (max_n - min_n)))
    return int(np.clip(N, min_n, max_n))

def prepare_sector_features_from_options(
    options_csv: str,
    etf_tickers: List[str],
    backtest_start: int,
    backtest_end: int,
    lookback_years: int,
    options_data_end_date: str,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:

    L.info("Options prep | filtering tickers/dates")
    date_min = _month_end(pd.Timestamp(f"{backtest_start}-01-01")) - pd.DateOffset(years=lookback_years)
    date_max = _month_end(pd.Timestamp(options_data_end_date))

    etf_tickers = [t.upper() for t in etf_tickers]
    ticker_to_sector = {k.upper(): v for k, v in SECTOR_ETFS.items()}
    use_cols = ["date","exdate","cp_flag","strike_price","volume","open_interest",
                "impl_volatility","delta","vega","ticker","forward_price"]

    all_chunks: List[pd.DataFrame] = []
    row_count = 0

    for chunk in pd.read_csv(options_csv, usecols=use_cols, chunksize=chunksize):
        df = chunk.copy()
        row_count += len(df)
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df[df["ticker"].isin(etf_tickers)]
        if df.empty:
            continue

        df["date"] = _month_end(pd.to_datetime(df["date"], errors="coerce"))
        df = df[(df["date"] >= date_min) & (df["date"] <= date_max)]
        if df.empty:
            continue

        df["exdate"] = pd.to_datetime(df["exdate"], errors="coerce")
        df["cp_flag"] = df["cp_flag"].astype(str).str.upper().str.strip()
        df = _safe_numeric(df, ["volume","open_interest","impl_volatility","delta","vega","forward_price","strike_price"])
        df["volume"] = df["volume"].fillna(0.0)
        df["open_interest"] = df["open_interest"].fillna(0.0)

        dte = (df["exdate"] - df["date"]).dt.days
        df = df[(dte >= 0) & (dte <= 720)]
        df["term_bucket"] = pd.cut(dte, bins=[-1,45,90,10_000], labels=["short","mid","long"])

        is_put = (df["cp_flag"]=="P").astype(float)
        is_call = (df["cp_flag"]=="C").astype(float)
        df["vol_put"] = df["volume"] * is_put
        df["vol_call"] = df["volume"] * is_call
        df["oi_put"] = df["open_interest"] * is_put
        df["oi_call"] = df["open_interest"] * is_call

        df["iv_mul_vol"] = df["impl_volatility"].fillna(0.0) * df["volume"]
        df["delta_mul_vol"] = df["delta"].fillna(0.0) * df["volume"]
        df["vega_tilt"] = np.sign(df["delta"].fillna(0.0)) * df["vega"].fillna(0.0)

        df["moneyness"] = df["strike_price"] / (df["forward_price"] + 1e-12)
        df["atm_volume"] = df["volume"] * ((df["moneyness"] >= 0.95) & (df["moneyness"] <= 1.05)).astype(float)

        df["month_end"] = df["date"] + pd.offsets.MonthEnd(0)
        gp = df.groupby(["month_end","ticker","term_bucket"], observed=True, dropna=False)
        core = gp[["volume","open_interest","vol_put","vol_call","oi_put","oi_call",
                   "iv_mul_vol","delta_mul_vol","vega_tilt","atm_volume"]].sum().reset_index()
        core["iv_vw"] = core["iv_mul_vol"] / (core["volume"] + 1e-12)
        core["delta_vw"] = core["delta_mul_vol"] / (core["volume"] + 1e-12)
        core = core.rename(columns={"volume":"vol_tot","open_interest":"oi_tot","month_end":"date"})

        def piv(df_term, col):
            p = df_term.pivot(index=["date","ticker"], columns="term_bucket", values=col)
            p.columns = [f"{col}_{c}" for c in p.columns]
            return p

        pieces = [piv(core, c) for c in ["vol_tot","oi_tot","vol_put","vol_call","oi_put","oi_call","iv_vw","delta_vw","vega_tilt","atm_volume"]]
        wide = pd.concat(pieces, axis=1).reset_index()

        def s(name: str) -> pd.Series:
            return wide[name] if name in wide.columns else pd.Series(0.0, index=wide.index)

        for term in ["short","mid","long"]:
            wide[f"pcr_vol_{term}"] = s(f"vol_put_{term}") / (s(f"vol_call_{term}") + 1e-12)
            wide[f"pcr_oi_{term}"]  = s(f"oi_put_{term}")  / (s(f"oi_call_{term}")  + 1e-12)
        wide["iv_slope_ts"] = s("iv_vw_long") - s("iv_vw_short")

        all_chunks.append(wide)

    if not all_chunks:
        L.warning("Options prep | nothing after filtering (read %d rows)", row_count)
        return pd.DataFrame(columns=["date","sector"])

    opt_ticker_month = pd.concat(all_chunks, ignore_index=True).drop_duplicates(subset=["date","ticker"])
    opt_ticker_month["sector"] = opt_ticker_month["ticker"].map(ticker_to_sector).str.lower()
    feats = (opt_ticker_month.dropna(subset=["sector"])
             .groupby(["date","sector"], as_index=False)
             .mean(numeric_only=True)
             .sort_values(["date","sector"]))
    L.info("Options coverage: %s .. %s | sectors=%d | rows=%d",
           str(feats["date"].min().date()), str(feats["date"].max().date()),
           feats["sector"].nunique(), len(feats))
    return feats

def fit_predict_sector_ridge(
    train_returns_wide: pd.DataFrame,
    feats_long: pd.DataFrame,
    target_date: pd.Timestamp,
    max_feature_lag_months: int = 3
) -> Optional[pd.Series]:

    if feats_long is None or feats_long.empty:
        return None

    Rw = train_returns_wide.copy()
    Rw.index = pd.to_datetime(Rw.index).to_period("M").to_timestamp("M")
    Rw = Rw.sort_index()
    sectors = list(Rw.columns)

    F = feats_long.copy()
    F["date"] = pd.to_datetime(F["date"], errors="coerce")
    F = F.dropna(subset=["date"])
    F["date"] = F["date"].dt.to_period("M").dt.to_timestamp("M")
    F = F.sort_values(["sector", "date"])

    T_end = pd.to_datetime(target_date).to_period("M").to_timestamp("M")
    cutoff_train = min(T_end - pd.offsets.MonthEnd(1), F["date"].max())

    rows = []
    for s in sectors:
        sr = Rw[s].sort_index()
        Xs = F.loc[F["sector"] == s].set_index("date").sort_index()
        if Xs.empty:
            continue
        sr_next = sr.shift(-1).rename("ret_next")
        aligned = pd.concat([sr_next, Xs], axis=1).dropna(subset=["ret_next"])
        aligned = aligned[aligned.index < cutoff_train]
        if aligned.empty:
            continue
        feats_only = (aligned.drop(columns=["ret_next"], errors="ignore")
                            .select_dtypes(include=[np.number])
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0.0))
        y_part = aligned["ret_next"].reindex(feats_only.index)
        for dt, xrow in feats_only.iterrows():
            rows.append({"sector": s, "date": dt, "y": float(y_part.loc[dt]), **xrow.to_dict()})

    if not rows:
        return None

    TR = pd.DataFrame(rows).sort_values(["sector", "date"])
    y = TR["y"].to_numpy()
    feature_cols = sorted(TR.drop(columns=["sector","date","y"]).select_dtypes(include=[np.number]).columns.tolist())
    if not feature_cols:
        return None

    X = (TR[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy())
    if X.size == 0:
        return None

    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=1.0).fit(scaler.transform(X), y)

    mu_pred = {}
    usable = 0
    Tm1 = T_end - pd.offsets.MonthEnd(1)

    for s in sectors:
        Fs = F.loc[F["sector"] == s].set_index("date").sort_index()
        if Fs.empty:
            continue
        Fs_Tm1 = Fs[Fs.index <= Tm1]
        if Fs_Tm1.empty:
            continue
        last_dt = Fs_Tm1.index[-1]
        lag_m = Tm1.to_period("M").ordinal - last_dt.to_period("M").ordinal
        if lag_m is None or lag_m > max_feature_lag_months:
            continue
        x_pred_df = (Fs_Tm1.loc[[last_dt]].reindex(columns=feature_cols)
                     .replace([np.inf, -np.inf], np.nan).fillna(0.0))
        mu_pred[s] = float(model.predict(scaler.transform(x_pred_df.to_numpy()))[0])
        usable += 1

    if usable == 0:
        return None
    return pd.Series({s: mu_pred.get(s, 0.0) for s in sectors})

def optimize_sector_weights(etf_returns_long: pd.DataFrame,
                            target_date: pd.Timestamp,
                            lookback_years: int,
                            decay_rate: Optional[float],
                            min_active_frac: float,
                            sector_features: Optional[pd.DataFrame],
                            lambda_forecast_default: float,
                            word_momentum_df: Optional[pd.DataFrame] = None,
                            word_momentum_weight: float = 0.1) -> Dict[str, float]:

    end_training = target_date - pd.offsets.MonthEnd(1)
    start_training = end_training - pd.DateOffset(years=lookback_years)
    train = etf_returns_long[(etf_returns_long['date'] >= start_training) & (etf_returns_long['date'] <= end_training)]
    R = train.pivot(index='date', columns='sector', values='monthly_return').sort_index().fillna(0.0)
    T, N = R.shape
    if T < 24:
        w = np.full(N, 1.0 / N)
        return dict(zip(R.columns, w))

    if decay_rate is None:
        decay_rate = estimate_decay_rate_from_training(R)
    tw = compute_exponential_weights(T, decay_rate)

    X = R.values
    mu = np.average(X, axis=0, weights=tw)
    Xc = X - mu
    W = np.diag(tw / tw.sum())
    cov = (Xc.T @ W @ Xc) / max(1e-8, (1.0 - float(tw @ tw)))
    cov += np.eye(N)*1e-8

    lambda_forecast = 0.0
    mu_pred = fit_predict_sector_ridge(R, sector_features, target_date, max_feature_lag_months=3)
    if mu_pred is not None:
        lambda_forecast = float(lambda_forecast_default)
        mu = (1.0 - lambda_forecast) * mu + lambda_forecast * mu_pred.reindex(R.columns).fillna(0.0).values
        L.info("%s | Options features used | lambda=%.2f | pred_range=[%.4f,%.4f]",
               (target_date - pd.offsets.MonthEnd(1)).date(), lambda_forecast,
               mu_pred.min(), mu_pred.max())
    else:
        L.info("%s | Options features not usable this month (lambda=0)",
               (target_date - pd.offsets.MonthEnd(1)).date())

    # Word momentum adjustment
    if MOMENTUM_ENABLED and word_momentum_df is not None and not word_momentum_df.empty:
        momentum_data = word_momentum_df[word_momentum_df['date'] == target_date]
        
        if not momentum_data.empty:
            momentum_adj = np.zeros(N)
            for i, sector in enumerate(R.columns):
                if sector in MOMENTUM_SECTORS:
                    z = momentum_data[momentum_data['sector'] == sector]['mom_6m']
                    if not z.empty and not pd.isna(z.iloc[0]):
                        momentum_adj[i] = z.iloc[0]
            momentum_adj = np.clip(momentum_adj, -0.1, 0.1) * MOMENTUM_WEIGHT
            mu = mu + momentum_adj
            L.info("%s | Word momentum applied to %d sectors (weight=%.2f, range=[%.4f,%.4f])",
                   (target_date - pd.offsets.MonthEnd(1)).date(), 
                   sum(1 for x in momentum_adj if x != 0), MOMENTUM_WEIGHT,
                   momentum_adj.min(), momentum_adj.max())

    w_min, w_max = compute_sector_bounds(N, min_active_frac) if IS_MAX_ALLOCATION else (0.0, 1.0)

    def neg_sharpe(w):
        r = w @ mu
        v = np.sqrt(max(1e-12, w @ cov @ w))
        return -r / v

    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bnds = [(w_min, w_max) for _ in range(N)]
    w0 = np.full(N, 1.0/N)
    res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': 500})
    w = res.x if (res.success and np.all(np.isfinite(res.x))) else w0
    w = np.clip(w, w_min, w_max); w = w / w.sum()
    return dict(zip(R.columns, w))

def build_optimized_long_with_contrib(g: pd.DataFrame,
                                      sector_weights: Dict[str,float],
                                      total_stocks: int) -> Tuple[float, Dict[str, Dict[str, float]], List[str]]:
    n_by_sector = allocate_sector_cardinality(sector_weights, total_stocks)
    picks = []
    for s, w in sector_weights.items():
        if w <= 0:
            continue
        sg = g[g["sector"] == s].dropna(subset=["pred","stock_ret"])
        if sg.empty:
            continue
        n_s = max(1, n_by_sector.get(s, 0))
        picks.append(sg.nlargest(min(n_s, len(sg)), "pred").assign(_sector=s, _sector_w=w))
    if not picks:
        return np.nan, {}, []
    sel = pd.concat(picks, ignore_index=True)
    counts = sel.groupby("_sector").size().to_dict()
    sel["_stock_w"] = sel.apply(lambda r: r["_sector_w"] / counts[r["_sector"]], axis=1)
    contrib = {}
    for s, w in sector_weights.items():
        sg = sel[sel["_sector"] == s]
        if sg.empty:
            continue
        avg_ret = float(sg["stock_ret"].mean())
        contrib[s] = {"weight": float(w), "avg_ret": avg_ret, "contrib": float(w*avg_ret), "n": int(len(sg))}
    total_ret = float(sum(v["contrib"] for v in contrib.values()))
    return total_ret, contrib, sel["id"].astype(str).tolist()

def turnover_rate(prev_ids: set, curr_ids: set) -> float:
    if not prev_ids:
        return 0.0
    return len(prev_ids.symmetric_difference(curr_ids)) / max(1, len(prev_ids))

def run_backtest(returns_data: pd.DataFrame,
                 backtest_start: int,
                 backtest_end: int,
                 lookback_years: int,
                 min_active_frac: float,
                 decay_rate: Optional[float],
                 sector_features: pd.DataFrame,
                 lambda_forecast: float,
                 total_names_bounds: Tuple[int,int] = (100,250),
                 word_momentum_df: Optional[pd.DataFrame] = None,
                 word_momentum_weight: float = 0.1) -> pd.DataFrame:

    E = fetch_etf_monthly(list(SECTOR_ETFS.keys()))
    etf_long = get_monthly_etf_returns_from_pivot(E)

    ssi_df = load_ssi(SSI_CSV)           
    hmm_df = load_hmm_states(HMM_CSV)

    m0 = _month_end(pd.Timestamp(f"{backtest_start}-01-01"))
    m1 = _month_end(pd.Timestamp(f"{backtest_end}-12-31"))
    months = (returns_data[(returns_data["date"] >= m0) & (returns_data["date"] <= m1)]["date"].sort_values().unique())
    
    # SHIFT EVERYTHING BY 1 MONTH TO AVOID LOOKAHEAD BIAS
    sector_features = sector_features.copy()
    sector_features["date"] = pd.to_datetime(sector_features["date"]).dt.to_period("M").dt.to_timestamp("M") + pd.offsets.MonthEnd(1)

    etf_long = etf_long.copy()
    etf_long["date"] = pd.to_datetime(etf_long["date"]).dt.to_period("M").dt.to_timestamp("M") + pd.offsets.MonthEnd(1)

    ssi_df = ssi_df.copy()
    ssi_df["date"] = pd.to_datetime(ssi_df["date"]).dt.to_period("M").dt.to_timestamp("M") + pd.offsets.MonthEnd(1)

    hmm_df = hmm_df.copy()
    hmm_df["date"] = pd.to_datetime(hmm_df["date"]).dt.to_period("M").dt.to_timestamp("M") + pd.offsets.MonthEnd(1)

    prev_SSI_long = prev_SSI_short = set()
    to_SSI = []
    rows = []
    
    # Track previous SSI allocation for smoothing
    prev_ssi_long_alloc: Optional[float] = None
    prev_ssi_short_alloc: Optional[float] = None
    

    for i, dt in enumerate(months, 1):
        print(f"Month {i} of {len(months)}: {dt.date()}")
        g = returns_data[returns_data["date"] == dt].dropna(subset=["pred","stock_ret","sector"]).copy()
        if g.empty:
            continue
        # z-score per (date, sector)
        g["pred_sec"] = g.groupby("sector")["pred"].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        )
        
        # Only optimize sector weights for SSI fallback case
        opt_w = None
        contrib_ssi = None
        ssi_current = ssi_df[ssi_df["date"] == dt]
        hmm_current = hmm_df[hmm_df["date"] == dt]

        # SSI/HMM split
        if not ssi_current.empty and not hmm_current.empty:
            ssi_value = float(ssi_current.filter(regex="^SSI$|^ssi$", axis=1).iloc[0, 0])
            ssi_d1 = float(ssi_current.get("ssi_d1", pd.Series([0.0])).iloc[0])
            ssi_d2 = float(ssi_current.get("ssi_d2", pd.Series([0.0])).iloc[0])
            ssi_i1 = float(ssi_current.get("ssi_i1", pd.Series([0.0])).iloc[0])
            ssi_i2 = float(ssi_current.get("ssi_i2", pd.Series([0.0])).iloc[0])
            hmm_state = int(hmm_current["state"].iloc[0])
            p_state_0 = float(hmm_current["p_state_0"].iloc[0])
            p_state_1 = float(hmm_current["p_state_1"].iloc[0])
            target_long_alloc, target_short_alloc = compute_ssi_hmm_allocation(ssi_value, ssi_d1, ssi_d2, ssi_i1, ssi_i2,
                                                                 hmm_state, p_state_0, p_state_1)
            # Apply smoothing to reduce turnover
            long_alloc, short_alloc = smooth_allocation(target_long_alloc, target_short_alloc,
                                                        prev_ssi_long_alloc, prev_ssi_short_alloc,
                                                        max_change_per_month=0.15, smoothing_alpha=0.7)
            prev_ssi_long_alloc = long_alloc
            prev_ssi_short_alloc = short_alloc
            
            N_total_ssi = dynamic_total_names_from_dispersion(g["pred"], base=125, min_n=total_names_bounds[0], max_n=total_names_bounds[1])
            nL_ssi = max(20, int(round(N_total_ssi * long_alloc)))
            nS_ssi = max(20, int(round(N_total_ssi * short_alloc)))
            if nL_ssi + nS_ssi > N_total_ssi: 
                if nL_ssi > nS_ssi:
                    nL_ssi = N_total_ssi - nS_ssi
                else:
                    nS_ssi = N_total_ssi - nL_ssi
            
            # Use turnover-constrained rebuild if we have a previous portfolio
            if prev_SSI_long or prev_SSI_short:
                # Compute dynamic max turnover based on SSI
                max_turnover_ssi = compute_max_turnover_from_ssi(ssi_value)
                LS_ret_ssi, LS_long_ids_ssi, LS_short_ids_ssi = rebuild_portfolio_with_turnover_constraint(
                    g, prev_SSI_long, prev_SSI_short, nL_ssi, nS_ssi, max_turnover=max_turnover_ssi, score_col="pred_sec"
                )
                # Log turnover
                long_turnover = len(prev_SSI_long - set(LS_long_ids_ssi)) / max(1, len(prev_SSI_long)) if prev_SSI_long else 0.0
                short_turnover = len(prev_SSI_short - set(LS_short_ids_ssi)) / max(1, len(prev_SSI_short)) if prev_SSI_short else 0.0
                L.info("SSI/HMM %s | SSI=%.3f state=%d (p1=%.2f) | target: long=%.2f short=%.2f | smoothed: long=%.2f short=%.2f | nL=%d nS=%d | max_TO=%.1f%% | long_TO=%.1f%% short_TO=%.1f%%",
                       dt.date(), ssi_value, hmm_state, p_state_1, target_long_alloc, target_short_alloc, long_alloc, short_alloc, 
                       nL_ssi, nS_ssi, max_turnover_ssi*100, long_turnover*100, short_turnover*100)
            else:
                # First month: build from scratch
                LS_ret_ssi, LS_long_ids_ssi, LS_short_ids_ssi = build_long_short_portfolio_sector_neutral(
                    g, nL_ssi, nS_ssi, sector_weights=None, score_col="pred_sec"
                )
                L.info("SSI/HMM %s | SSI=%.3f state=%d (p1=%.2f) | target: long=%.2f short=%.2f | smoothed: long=%.2f short=%.2f | nL=%d nS=%d | (first month)",
                       dt.date(), ssi_value, hmm_state, p_state_1, target_long_alloc, target_short_alloc, long_alloc, short_alloc, nL_ssi, nS_ssi)
        elif not ssi_current.empty:
            ssi_value = float(ssi_current.filter(regex="^SSI$|^ssi$", axis=1).iloc[0, 0])
            ssi_d1 = float(ssi_current.get("ssi_d1", pd.Series([0.0])).iloc[0])
            ssi_d2 = float(ssi_current.get("ssi_d2", pd.Series([0.0])).iloc[0])
            ssi_i1 = float(ssi_current.get("ssi_i1", pd.Series([0.0])).iloc[0])
            ssi_i2 = float(ssi_current.get("ssi_i2", pd.Series([0.0])).iloc[0])
            target_long_alloc, target_short_alloc = compute_ssi_allocation(ssi_value, ssi_d1, ssi_d2, ssi_i1, ssi_i2)
            # Apply smoothing to reduce turnover
            long_alloc, short_alloc = smooth_allocation(target_long_alloc, target_short_alloc,
                                                        prev_ssi_long_alloc, prev_ssi_short_alloc,
                                                        max_change_per_month=0.15, smoothing_alpha=0.7)
            prev_ssi_long_alloc = long_alloc
            prev_ssi_short_alloc = short_alloc
            
            N_total_ssi = dynamic_total_names_from_dispersion(g["pred"], base=175, min_n=total_names_bounds[0], max_n=total_names_bounds[1])
            nL_ssi = max(20, int(round(N_total_ssi * long_alloc)))
            nS_ssi = max(20, int(round(N_total_ssi * short_alloc)))
            if nL_ssi + nS_ssi > N_total_ssi:
                if nL_ssi > nS_ssi:
                    nL_ssi = N_total_ssi - nS_ssi
                else:
                    nS_ssi = N_total_ssi - nL_ssi
            
            # Use turnover-constrained rebuild if we have a previous portfolio
            if prev_SSI_long or prev_SSI_short:
                # Compute dynamic max turnover based on SSI
                max_turnover_ssi = compute_max_turnover_from_ssi(ssi_value)
                LS_ret_ssi, LS_long_ids_ssi, LS_short_ids_ssi = rebuild_portfolio_with_turnover_constraint(
                    g, prev_SSI_long, prev_SSI_short, nL_ssi, nS_ssi, max_turnover=max_turnover_ssi, score_col="pred_sec"
                )
                # Log turnover
                long_turnover = len(prev_SSI_long - set(LS_long_ids_ssi)) / max(1, len(prev_SSI_long)) if prev_SSI_long else 0.0
                short_turnover = len(prev_SSI_short - set(LS_short_ids_ssi)) / max(1, len(prev_SSI_short)) if prev_SSI_short else 0.0
                L.info("SSI only %s | SSI=%.3f | nL=%d nS=%d | max_TO=%.1f%% | long_TO=%.1f%% short_TO=%.1f%%",
                       dt.date(), ssi_value, nL_ssi, nS_ssi, max_turnover_ssi*100, long_turnover*100, short_turnover*100)
            else:
                # First month: build from scratch
                LS_ret_ssi, LS_long_ids_ssi, LS_short_ids_ssi = build_long_short_portfolio_sector_neutral(
                    g, nL_ssi, nS_ssi, sector_weights=None, score_col="pred_sec"
                )
                L.info("SSI only %s | SSI=%.3f | nL=%d nS=%d | (first month)", dt.date(), ssi_value, nL_ssi, nS_ssi)
        else:
            # Fallback: optimized long-only (only compute opt_w if needed)
            if opt_w is None:
                L.info("[%4d/%4d] %s | optimizing sector weights for fallback", i, len(months), dt.date())
                opt_w = optimize_sector_weights(etf_long, dt, lookback_years, decay_rate, min_active_frac,
                                                sector_features, lambda_forecast, word_momentum_df, word_momentum_weight)
            N_total_ssi = dynamic_total_names_from_dispersion(g["pred"], base=175, min_n=total_names_bounds[0], max_n=total_names_bounds[1])
            LS_ret_ssi, contrib_ssi, LS_long_ids_ssi = build_optimized_long_with_contrib(g, opt_w, N_total_ssi)
            LS_short_ids_ssi = []
            nL_ssi, nS_ssi = len(LS_long_ids_ssi), 0
            L.warning("No SSI data %s -> fallback long-only", dt.date())
        
        # Turnover
        to_SSI.append(turnover_rate(prev_SSI_long | prev_SSI_short, set(LS_long_ids_ssi) | set(LS_short_ids_ssi)))
        prev_SSI_long, prev_SSI_short = set(LS_long_ids_ssi), set(LS_short_ids_ssi)

        row = {
            "date": dt, "year": dt.year,
            "port_SSI": LS_ret_ssi,
            "nL_ssi": nL_ssi, "nS_ssi": nS_ssi,
        }
        # Only add sector weights and contributions if opt_w was computed (fallback case)
        if opt_w is not None:
            for s, w in opt_w.items():
                row[f"weight_{s}"] = w
        if contrib_ssi is not None:
            for s, d in contrib_ssi.items():
                row[f"contrib_{s}"] = d["contrib"]
                row[f"secavg_{s}"] = d["avg_ret"]
                row[f"n_{s}"] = d["n"]
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("date")
    results["turnover_SSI"] = pd.Series(to_SSI, index=results.index)
    
    return results

def load_benchmark_data(benchmark_file: str) -> pd.Series:
    """Wrapper for config.load_benchmark_data to maintain compatibility."""
    return load_benchmark_data_config(benchmark_file)

def load_mkt_excess(mkt_file: str) -> pd.Series:
    """Load market excess returns for CAPM calculations."""
    mkt = pd.read_csv(mkt_file)
    if {"year","month","ret","rf"}.issubset(mkt.columns):
        dt = pd.to_datetime(mkt[["year","month"]].assign(day=1)) + pd.offsets.MonthEnd(0)
        ex = pd.to_numeric(mkt["ret"], errors="coerce") - pd.to_numeric(mkt["rf"], errors="coerce")
        return pd.Series(ex.values, index=dt, name="mkt_rf").sort_index()
    raise ValueError(f"{mkt_file} must have columns (year,month,ret,rf).")

# compute_capm_metrics and compute_portfolio_metrics are now imported from config

def create_portfolio_plots(results: pd.DataFrame, out_file: Path, benchmark_file: str) -> None:
    if results.empty:
        L.warning("No results to plot"); return
    benchmark = load_benchmark_data(benchmark_file)
    results['date'] = pd.to_datetime(results['date'])
    results = results.sort_values('date')

    portfolio_cols = {}
    # base
    if 'port_SSI' in results.columns:
        portfolio_cols['SSI-Based Portfolio'] = 'port_SSI'

    for name, col in portfolio_cols.items():
        if col not in results.columns:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{name} vs Benchmark', fontsize=16, fontweight='bold')

        port = results.set_index('date')[col].dropna()
        common = port.index.intersection(benchmark.index)
        port_al, bench_al = port.reindex(common), benchmark.reindex(common)
        if len(common) == 0:
            plt.close(); continue

        # Cum returns
        ax1 = axes[0, 0]
        pcum = (1 + port_al).cumprod(); bcum = (1 + bench_al).cumprod()
        ax1.plot(pcum.index, pcum, label=name, linewidth=3, color='blue')
        ax1.plot(bcum.index, bcum, label='Benchmark', linewidth=3, color='red', linestyle='--')
        ax1.set_title('Cumulative Returns', fontweight='bold'); ax1.set_ylabel('Cumulative Return')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); ax1.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Scatter
        ax2 = axes[0, 1]
        ax2.scatter(bench_al, port_al, alpha=0.6, s=30)
        mn, mx = min(bench_al.min(), port_al.min()), max(bench_al.max(), port_al.max())
        ax2.plot([mn, mx], [mn, mx], 'r--', alpha=0.5, label='Perfect Correlation')
        corr = port_al.corr(bench_al)
        ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax2.set_title('Monthly Returns Scatter', fontweight='bold')
        ax2.set_xlabel('Benchmark Return'); ax2.set_ylabel(f'{name} Return')
        ax2.legend(); ax2.grid(True, alpha=0.3)

        # Rolling Sharpe
        ax3 = axes[1, 0]
        if len(port_al) >= 12:
            pr = port_al.rolling(12)
            br = bench_al.rolling(12)
            rs_p = pr.mean() / pr.std() * np.sqrt(12)
            rs_b = br.mean() / br.std() * np.sqrt(12)
            ax3.plot(port_al.index, rs_p, label=name, linewidth=2, color='blue')
            ax3.plot(bench_al.index, rs_b, label='Benchmark', linewidth=2, color='red', linestyle='--')
        ax3.set_title('12-Month Rolling Sharpe Ratio', fontweight='bold'); ax3.set_ylabel('Sharpe')
        ax3.legend(); ax3.grid(True, alpha=0.3); ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); ax3.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

        # Drawdowns
        ax4 = axes[1, 1]
        pcum = (1 + port_al).cumprod(); prmax = pcum.expanding().max(); pdd = (pcum - prmax)/prmax
        bcum = (1 + bench_al).cumprod(); brmax = bcum.expanding().max(); bdd = (bcum - brmax)/brmax
        ax4.fill_between(port_al.index, pdd, 0, alpha=0.6, label=name, color='blue')
        ax4.fill_between(bench_al.index, bdd, 0, alpha=0.6, label='Benchmark', color='red')
        ax4.set_title('Drawdown Comparison', fontweight='bold'); ax4.set_ylabel('Drawdown')
        ax4.legend(); ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); ax4.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

        # Metrics box
        port_ann_ret = (1 + port_al.mean()) ** 12 - 1
        port_ann_vol = port_al.std() * np.sqrt(12)
        port_mdd = pdd.min()
        bench_ann_ret = (1 + bench_al.mean()) ** 12 - 1
        bench_ann_vol = bench_al.std() * np.sqrt(12)
        sharpe_p = port_ann_ret / port_ann_vol if port_ann_vol > 0 else 0
        sharpe_b = bench_ann_ret / bench_ann_vol if bench_ann_vol > 0 else 0
        
        # Calculate Sortino ratios
        port_downside = port_al[port_al < 0].std() * np.sqrt(12)
        bench_downside = bench_al[bench_al < 0].std() * np.sqrt(12)
        sortino_p = port_ann_ret / port_downside if port_downside > 0 else 0
        sortino_b = bench_ann_ret / bench_downside if bench_downside > 0 else 0
        
        info = (port_ann_ret - bench_ann_ret) / ((port_al - bench_al).std() * np.sqrt(12)) if (port_al - bench_al).std() > 0 else np.nan
        txt = (f"Performance Metrics:\n\n{name}:\n"
               f"  Ann. Return: {port_ann_ret:.1%}\n"
               f"  Ann. Vol: {port_ann_vol:.1%}\n"
               f"  Sharpe: {sharpe_p:.2f}\n"
               f"  Sortino: {sortino_p:.2f}\n"
               f"  Max DD: {port_mdd:.1%}\n\n"
               f"Benchmark:\n"
               f"  Ann. Return: {bench_ann_ret:.1%}\n"
               f"  Ann. Vol: {bench_ann_vol:.1%}\n"
               f"  Sharpe: {sharpe_b:.2f}\n"
               f"  Sortino: {sortino_b:.2f}\n\n"
               f"Excess Return: {port_ann_ret - bench_ann_ret:.1%}\n"
               f"Information Ratio: {info:.2f}")
        ax1.text(0.02, 0.98, txt, transform=ax1.transAxes, fontsize=9,
                 va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        plot_file = out_file.with_name(f"{out_file.stem}_{safe_name}_vs_benchmark.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight'); plt.close()
        L.info("Saved %s", plot_file)

def main():
    L.info("Loading predictions: %s", RETURNS_CSV)
    df = load_returns_data(RETURNS_CSV, "date")
    L.info("Pred rows: %d | %s .. %s", len(df), str(df["date"].min().date()), str(df["date"].max().date()))

    sector_feats = prepare_sector_features_from_options(
        options_csv=OPTIONS_CSV,
        etf_tickers=list(SECTOR_ETFS.keys()),
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
        lookback_years=LOOKBACK_YEARS,
        options_data_end_date=OPTIONS_DATA_END_DATE,
        chunksize=1_000_000,
    )

    L.info("Loading word momentum…")
    word_momentum_df = load_word_momentum(PATH_TO_WORD_MOMENTUM)

    L.info("Backtest %d–%d | lookback=%dyr", BACKTEST_START, BACKTEST_END, LOOKBACK_YEARS)
    results = run_backtest(
        returns_data=df,
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
        lookback_years=LOOKBACK_YEARS,
        min_active_frac=MIN_ACTIVE_FRAC,
        decay_rate=DECAY_RATE,
        sector_features=sector_feats,
        lambda_forecast=LAMBDA_FORECAST,
        total_names_bounds=(100, 250),
        word_momentum_df=word_momentum_df,
        word_momentum_weight=MOMENTUM_WEIGHT,
    )


    out = Path(OUT_FILE); out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False); L.info("Saved results to %s", out)

    # Plots
    L.info("Creating performance plots…")
    create_portfolio_plots(results, out, benchmark_file=MKT_FILE)

    # Metrics
    names_cols = {
        "SSI-Based Portfolio": "port_SSI",
    }

    turnovers = {
        "port_SSI": results.get("turnover_SSI"),
    }

    # Load benchmark for comparison
    benchmark = None
    try:
        benchmark = load_benchmark_data(MSCI_BENCHMARK_FILE)
        L.info("Loaded MSCI World Index benchmark returns from %s", MSCI_BENCHMARK_FILE)
    except Exception as e:
        L.warning(f"Could not load benchmark from {MSCI_BENCHMARK_FILE}: {e}")

    for name, col in names_cols.items():
        if col not in results.columns:
            continue
            
        L.info("=" * 80)
        L.info("PORTFOLIO: %s", name)
        L.info("=" * 80)
        
        # Get gross returns
        gross_returns = results.set_index('date')[col].dropna()
        turnover_series = turnovers.get(col)
        
        # Compute net returns using trading costs from PDF specifications
        # Trading costs: Large Cap (>= $15B) = 0.25% per turnover, Small Cap (< $15B) = 1.0% per turnover
        # Note: Using estimated cap composition. For accurate costs, compute actual cap composition
        # from holdings data using compute_cap_composition() function from config
        large_cap_fraction = 0.5  # Estimated: adjust based on actual portfolio composition if available
        if turnover_series is not None:
            # Align turnover with returns (turnover_series should be aligned with results DataFrame)
            # Convert to Series if it's not already
            if isinstance(turnover_series, pd.Series):
                turnover_aligned = turnover_series.reindex(gross_returns.index).fillna(0.0)
            else:
                # If it's not a Series, try to create one from results
                turnover_col = col.replace("port_", "turnover_")
                if turnover_col in results.columns:
                    turnover_aligned = results.set_index('date')[turnover_col].reindex(gross_returns.index).fillna(0.0)
                else:
                    turnover_aligned = pd.Series(0.0, index=gross_returns.index)
            net_returns = compute_net_returns(gross_returns, turnover_aligned, large_cap_fraction)
        else:
            net_returns = None
        
        # Compute GROSS metrics
        L.info("\n--- GROSS RETURNS METRICS ---")
        gross_metrics = compute_portfolio_metrics(gross_returns, turnover_series, prefix="Gross")
        for key, val in gross_metrics.items():
            if "Return" in key or "Volatility" in key or "Drawdown" in key or "Loss" in key:
                L.info(f"  {key}: {100*val:.2f}%")
            elif "Ratio" in key or "Sharpe" in key or "Information" in key:
                L.info(f"  {key}: {val:.3f}")
            elif "Turnover" in key:
                L.info(f"  {key}: {100*val:.2f}%")
            else:
                L.info(f"  {key}: {val:.4f}")
        
        # CAPM metrics for GROSS
        L.info("\n--- GROSS RETURNS CAPM METRICS ---")
        gross_capm = compute_capm_metrics(gross_returns, mkt_file=MKT_FILE, prefix="Gross")
        L.info(f"  Gross Alpha (monthly): {gross_capm.get('Gross Alpha (monthly)', np.nan):.4f}")
        L.info(f"  Gross Alpha (annual): {100*gross_capm.get('Gross Alpha (annual)', np.nan):.2f}%")
        L.info(f"  Gross Beta: {gross_capm.get('Gross Beta', np.nan):.3f}")
        L.info(f"  Gross Information Ratio: {gross_capm.get('Gross Information Ratio', np.nan):.3f}")
        L.info(f"  Gross Alpha t-stat (HAC): {gross_capm.get('Gross Alpha t-stat (HAC)', np.nan):.2f}")
        
        # Compute NET metrics if available
        if net_returns is not None:
            L.info("\n--- NET RETURNS METRICS (After Trading Costs) ---")
            net_metrics = compute_portfolio_metrics(net_returns, turnover_series, prefix="Net")
            for key, val in net_metrics.items():
                if "Return" in key or "Volatility" in key or "Drawdown" in key or "Loss" in key:
                    L.info(f"  {key}: {100*val:.2f}%")
                elif "Ratio" in key or "Sharpe" in key or "Information" in key:
                    L.info(f"  {key}: {val:.3f}")
                elif "Turnover" in key:
                    L.info(f"  {key}: {100*val:.2f}%")
                else:
                    L.info(f"  {key}: {val:.4f}")
            
            # CAPM metrics for NET
            L.info("\n--- NET RETURNS CAPM METRICS (After Trading Costs) ---")
            net_capm = compute_capm_metrics(net_returns, mkt_file=MKT_FILE, prefix="Net")
            L.info(f"  Net Alpha (monthly): {net_capm.get('Net Alpha (monthly)', np.nan):.4f}")
            L.info(f"  Net Alpha (annual): {100*net_capm.get('Net Alpha (annual)', np.nan):.2f}%")
            L.info(f"  Net Beta: {net_capm.get('Net Beta', np.nan):.3f}")
            L.info(f"  Net Information Ratio: {net_capm.get('Net Information Ratio', np.nan):.3f}")
            L.info(f"  Net Alpha t-stat (HAC): {net_capm.get('Net Alpha t-stat (HAC)', np.nan):.2f}")
        
        # P&L Evolution
        L.info("\n--- P&L EVOLUTION (Starting AUM: $%dM) ---", INITIAL_AUM / 1e6)
        pnl_gross = compute_pnl_evolution(gross_returns, INITIAL_AUM)
        L.info(f"  Final AUM (Gross): ${pnl_gross.iloc[-1]/1e6:.2f}M")
        L.info(f"  Total Return (Gross): {100*(pnl_gross.iloc[-1]/INITIAL_AUM - 1):.2f}%")
        if net_returns is not None:
            pnl_net = compute_pnl_evolution(net_returns, INITIAL_AUM)
            L.info(f"  Final AUM (Net): ${pnl_net.iloc[-1]/1e6:.2f}M")
            L.info(f"  Total Return (Net): {100*(pnl_net.iloc[-1]/INITIAL_AUM - 1):.2f}%")
        
        # Turnover reporting
        if turnover_series is not None:
            L.info("\n--- TURNOVER (Monthly) ---")
            avg_turnover = turnover_series.mean()
            L.info(f"  Average Monthly Turnover: {100*avg_turnover:.2f}% (Target: {100*TARGET_TURNOVER:.0f}% per month)")
            L.info(f"  Note: Using stock-count method. PDF specifies weight-based formula:")
            L.info(f"        Turnover_t = (1/2) * sum_i |w_i,t - w_i,t-1|")
            if avg_turnover > TARGET_TURNOVER:
                L.warning(f"  WARNING: Monthly turnover exceeds target by {100*(avg_turnover - TARGET_TURNOVER):.2f}%")
        
        # Note: Cap composition, sector exposures, and country exposures would require
        # holdings data with weights. This would need to be added to the backtest results.
        L.info("\n--- COMPOSITION METRICS ---")
        L.info("  Note: Cap composition, sector exposures, and country exposures")
        L.info("  require holdings data with weights. Please add holdings tracking")
        L.info("  to backtest results for full reporting.")
        
        L.info("")

    L.info("=" * 80)
    L.info("Done.")

if __name__ == "__main__":
    main()

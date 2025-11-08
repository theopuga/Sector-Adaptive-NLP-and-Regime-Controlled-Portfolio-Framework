# Stochastic Pirates: Sector-Adaptive NLP and Regime-Controlled Portfolio Framework  

> *“Bridging financial linguistics, quantitative modeling, and adaptive control to understand how narratives shape markets.”*  

---

## Research Statement  

This repository presents a **research-grade framework** exploring the intersection of **Natural Language Processing (NLP)**, **macroeconomic clustering**, and **dynamic portfolio optimization**.  
Developed by **Stochastic Pirates**, the project investigates how **progressive word-momentum dictionaries**, **FinBERT-based sentiment**, and **quantile-based predictive modeling** can generate persistent alpha across market regimes.  

Originally designed for the **McGill FIAM 2025 Asset Management Competition**, the system has since evolved into an **independent quantitative research pipeline** emphasizing rigor, causality, and interpretability.

---

## Overview  

Our core objective:  
> **Predict sector-level returns through dynamic narrative signals derived from financial text** — blending firm-level sentiment, word-frequency evolution, and cross-sector market structure.  

The framework integrates:  
1. **FinBERT sentiment scoring** on 10-K / 10-Q filings.  
2. **Progressive word-momentum tracking** with exponential decay.  
3. **Sector clustering** via ETF correlations and soft probabilities.  
4. **Quantile XGBoost regressors** trained with rolling 10-year / 1-year OOS windows.  
5. **Regime control** through a Hidden Markov Model (HMM) and PID-based exposure modulation.  

---

## Methodology  

### 1. FinBERT Sentiment Extraction  
Each SEC filing is parsed sentence-by-sentence using the FinBERT model (`sec_finbert_monthly_optimized.py`).  
Outputs are aggregated into monthly firm-level sentiment factors:  
\[
\text{Sentiment}_{i,t} = P(\text{pos})_{i,t} - P(\text{neg})_{i,t}
\]
These scores are merged with Compustat mappings for global coverage.

---

### 2. Progressive Word-Momentum Dictionaries  
The dictionary expands as new sector-specific terms emerge (e.g., *“AI chip”*, *“EV battery”*).  
Words are activated 12 months after first observation to avoid forward bias (`map_new_words_to_sectors.py`, `validate_activation_map.py`).  

**Exponential Decay Model:**  
\[
v_t = \alpha x_t + (1 - \alpha)v_{t-1}
\]
where  
- \(x_t\) = raw word frequency share,  
- \(v_t\) = decayed trend,  
- \(\alpha\) ≈ 0.3 controls half-life memory.  

This ensures weakening sector correlations fade naturally while persistent narratives remain.

---

### 3. Sector Clustering and ETF Mapping  
`create_sector.py` and `corr_etfs.py` compute pairwise correlations between securities and sector ETFs (e.g., XLE, XLK, EEM).  
Soft-probability cluster assignments guide sector-level model training.

---

### 4. Quantile XGBoost Regression  
Each sector runs an independent quantile XGBoost pipeline (`xgboost_model_uncertainty.py`):  
- Rolling 10 y train / 1 y validation windows.  
- GPU-accelerated `hist` tree method.  
- Feature selection via in-sample importance ranking (top K ≈ 100).  

Outputs include quantile-predicted returns \(y_{p05},\,y_{p50},\,y_{p95}\) for uncertainty estimation.

---

### 5. Regime-Adaptive Portfolio Control  

#### Hidden Markov Model (HMM)  
Detects macro regimes (bull, bear, neutral) using option-implied sentiment, volatility spreads, and market breadth (`hmm_ssi.py`).  

#### PID Exposure Controller  
Exposure \(E_t\) dynamically adjusts with tracking error \(e_t\):  
\[
E_t = K_p e_t + K_i \int_0^t e_\tau\,d\tau + K_d \frac{de_t}{dt}
\]
with empirically tuned \((K_p, K_i, K_d)\) ensuring smooth volatility targeting and rapid regime adaptation.  

Implemented in `portfolio.py`.

---

## Key Findings  

1. **Narrative Persistence Predicts Sector Momentum** – sectors exhibiting rising word-momentum in filings outperform over the following 1–3 months.  
2. **Decay Filtering Improves Signal Stability** – exponential half-life smoothing increases Sharpe ≈ +0.3 vs raw counts.  
3. **Cross-Regime Robustness** – FinBERT sentiment loses strength during crises, while word-momentum remains predictive.  
4. **Regime-Controlled Exposure** – the PID + HMM controller maintains volatility < 30 % and drawdowns < 20 %.  
5. **Best Alpha Drivers** – Technology, Energy, and Emerging Markets contributed the most consistent excess returns.  

---

## Performance Summary  

| Metric | Value |
|:--|--:|
| **Net Annualized Return** | 58.43 % |
| **Alpha (vs MSCI World)** | +49.7 % |
| **Beta** | – 0.25 |
| **Sharpe Ratio** | 2.30 |
| **Sortino Ratio** | 2.55 |
| **Information Ratio** | 2.47 |
| **Max Drawdown** | – 39.3 % |
| **Hit Ratio** | 0.82 |

**Figures**  
- `fig1_sentiment_regimes.png` – Sentiment & Regime Alignment  
- `fig2_word_momentum_vs_market.png` – Sector Momentum vs MSCI World  
- `fig3_portfolio_vs_benchmark.png` – Strategy Equity Curve  
- `fig4_feature_importance.png` – Top Predictive Features  

---

## 🏗 Implementation Architecture  

| Script | Description |
|:--|:--|
| `combine_shards_to_filings_clean.py` | Merges FinBERT outputs into monthly firm datasets |
| `make_sec_features.py` | Builds lagged & normalized financial features |
| `make_sector_momentum_progressive.py` | Constructs progressive dictionary momentum with decay |
| `freeze_sector_dictionary.py` | Freezes validated word–sector mappings |
| `xgboost_model_uncertainty.py` | Quantile XGBoost model training + uncertainty estimates |
| `portfolio.py` | Regime-adaptive portfolio optimizer |
| `hmm_ssi.py` | Hidden Markov Model for regime inference |
| `breadth_sp500.py` | Computes breadth and sentiment divergence |
| `add_decay_feature.py` | Applies exponential decay to term frequencies |
| `merge_predictions_clean.py` | Consolidates all sector predictions |

---

## ⚙️ Quick Setup  

```bash
git clone https://github.com/<yourusername>/stochastic-pirates.git
cd stochastic-pirates
pip install -r requirements.txt


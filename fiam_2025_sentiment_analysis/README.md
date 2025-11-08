# SEC Filing Sentiment Analysis Pipeline

Complete pipeline for processing SEC filings, linking to company identifiers, and computing monthly FinBERT sentiment scores.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Pipeline Steps](#pipeline-steps)
4. [Hardware Configuration](#hardware-configuration)
5. [Script Details](#script-details)
6. [Troubleshooting](#troubleshooting)
7. [Output Files](#output-files)

---

## Overview

This pipeline processes raw SEC filing data through three main stages:

1. **Feature Extraction** (`make_sec_features.py`): Converts raw filings into standardized parquet shards
2. **Company Linking** (`link_global_shards.py`): Links filings to company identifiers (gvkey, iid) using name merge tables
3. **Sentiment Analysis** (`sec_finbert_monthly_optimized.py`): Computes FinBERT sentiment scores and aggregates to monthly time series

---

## Prerequisites

### Software Requirements

```bash
# Python 3.11+ with conda/miniconda
conda create -n fiam-2025 python=3.11
conda activate fiam-2025

# Core dependencies
pip install pandas pyarrow polars numpy scipy
pip install transformers torch
pip install pyyaml

# For GPU support (optional but recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Data Requirements

Before running, ensure you have:

1. **Raw SEC filing data** in `data/textsec/raw/US/` and `data/textsec/raw/Global/`
   - Supported formats: `.pkl`, `.parquet`, `.csv`, `.json`, `.jsonl`
   
2. **Name merge CSV files** for company linking:
   - `data/raw/Global_Name_Merge_by_DataDate_GVKEY_IID.csv`
   - `data/raw/North_America_Company_Merge_by_DataDate_GVKEY_IID.csv`
   
3. **Configuration file**: `config/paths.yaml` (see example below)

### Directory Structure

```
fiam_2025_sentiment_analysis/
├── config/
│   └── paths.yaml                 # Path configuration
├── data/
│   ├── raw/
│   │   ├── Global_Name_Merge_by_DataDate_GVKEY_IID.csv
│   │   └── North_America_Company_Merge_by_DataDate_GVKEY_IID.csv
│   └── textsec/
│       ├── raw/
│       │   ├── US/                # US filing files (by year)
│       │   └── Global/             # Global filing files (by country code)
│       ├── interim/
│       │   ├── filings_clean_shards/           # Fused shards (all countries + US)
│       │   ├── filings_clean_shards_us_split/   # US split shards (rf/mgmt preserved)
│       │   └── filings_clean_manifest.csv       # Manifest of all shards
│       └── processed/
│           └── FinBert/
│               ├── checkpoints/    # Per-shard checkpoint files
│               └── finbert_master.parquet      # Final output
└── run/
    ├── make_sec_features.py
    ├── link_global_shards.py
    └── sec_finbert_monthly_optimized.py
```

### Example `config/paths.yaml`

```yaml
textsec:
  raw_dir: "data/textsec/raw"
  raw_dir_us: "data/textsec/raw/US"
  raw_dir_global: "data/textsec/raw/Global"
  filings_clean_shards: "data/textsec/interim/filings_clean_shards"
  filings_clean_manifest: "data/textsec/interim/filings_clean_manifest.csv"
  finbert_master: "data/textsec/processed/FinBert/finbert_master.parquet"
```

---

## Pipeline Steps

### Step 1: Extract and Clean Features

**Script**: `run/make_sec_features.py`

Converts raw filing data into standardized parquet shards with consistent schema.

**Command**:
```bash
python run/make_sec_features.py \
  --paths config/paths.yaml \
  --num_workers 16 \
  --us_num_workers 2
```

**What it does**:
- Scans `data/textsec/raw/US/` and `data/textsec/raw/Global/` for filing files
- Normalizes column names (company_name, filing_date, form_type, etc.)
- For US files: creates TWO outputs:
  - **Fused shards**: Single `text` column (rf + mgmt merged)
  - **Split shards**: Separate `rf` and `mgmt` columns (no `text`)
- For Global files: creates fused shards only
- Writes parquet files organized by country in shard directories
- Creates manifest CSV tracking all output shards

**Output**:
- `data/textsec/interim/filings_clean_shards/` (fused shards)
- `data/textsec/interim/filings_clean_shards_us_split/` (US split shards)
- `data/textsec/interim/filings_clean_manifest.csv` (manifest)

**Key Parameters**:
- `--num_workers`: Workers for global files (default: min(8, cpu_count))
- `--us_num_workers`: Workers for US files (default: min(2, cpu_count)) - reduced due to memory pressure
- `--debug`: Enable verbose logging

**Performance Notes**:
- US files use fewer workers (default 2) to avoid memory issues with large pickle files
- Files that fail in parallel are automatically retried sequentially
- Processing is shard-based for memory efficiency

---

### Step 2: Link to Company Identifiers

**Script**: `run/link_global_shards.py`

Links filings to company identifiers (gvkey, iid) using name merge tables.

**Command**:
```bash
python run/link_global_shards.py \
  --paths config/paths.yaml \
  --name_merge "data/raw/Global_Name_Merge_by_DataDate_GVKEY_IID.csv" \
  --extra_name_merge "data/raw/North_America_Company_Merge_by_DataDate_GVKEY_IID.csv" \
  --company_col conm \
  --country_col fic \
  --date_col datadate \
  --gvkey_col gvkey \
  --iid_col iid \
  --num_workers 4
```

**What it does**:
- Reads shard manifest from Step 1
- Loads and unions name merge CSV files
- For each shard, attempts to link company names to gvkey/iid using:
  1. **Exact matching** (3 tiers):
     - Tier 1: company_norm + country + exact date
     - Tier 2: company_norm + country + same year
     - Tier 3: company_norm + country
  2. **Fuzzy matching** (for remaining rows):
     - Constrained by country and year buckets
     - Uses string similarity (SequenceMatcher)
     - Special handling for CAN, HKG, CHN markets
- Updates shards in-place with gvkey/iid values
- Writes atomically (temp file → rename) to prevent corruption

**Output**:
- Updates shards in `data/textsec/interim/filings_clean_shards/` with gvkey/iid columns

**Key Parameters**:
- `--name_merge`: Primary name merge CSV (required)
- `--extra_name_merge`: Additional name merge CSV(s) (optional, can repeat)
- `--company_col`: Company name column in CSV (e.g., `conm`)
- `--country_col`: Country/ISO column (e.g., `fic`)
- `--date_col`: Date column (e.g., `datadate`)
- `--gvkey_col`: GVKEY column (e.g., `gvkey`)
- `--iid_col`: IID column (e.g., `iid`)
- `--num_workers`: Parallel workers (default: min(4, cpu_count)) - reduced due to large merge tables
- `--countries`: Filter to specific countries (e.g., `"CAN,HKG,CHN"`)

**Performance Optimizations**:
- Limits fuzzy matching to 20,000 rows per shard (for very large shards like HK)
- Limits candidate pools to 2,000-5,000 candidates per row
- Automatically retries failed shards sequentially
- Atomic writes prevent corruption on crashes

**Linking Statistics**:
- Shows summary of linked rows per shard
- Reports total newly linked rows across all shards

---

### Step 3: FinBERT Sentiment Analysis

**Script**: `run/sec_finbert_monthly_optimized.py`

Computes FinBERT sentiment scores and aggregates to monthly time series.

**Command**:
```bash
python run/sec_finbert_monthly_optimized.py \
  --paths config/paths.yaml \
  --manifest "data/textsec/interim/filings_clean_manifest.csv" \
  --device cuda \
  --batch_size 64 \
  --num_workers 4 \
  --fp16
```

**What it does**:
- Reads shard manifest or discovers shards from directory
- For each shard:
  - Handles US format (rf/mgmt columns) vs global format (text column)
  - Normalizes company identifiers (gvkey, iid) and dates
  - Filters documents with valid text (length > 0)
  - Scores each document using FinBERT model
  - Aggregates to monthly level per (gvkey, iid, month_end, country, continent, section)
- For USA scope: creates both per-section and total aggregations
- Merges all monthly results
- Computes SEC sentiment features (momentum, moving averages, volatility, etc.)
- Writes final master file

**Output**:
- `data/textsec/processed/FinBert/checkpoints/*__monthly.parquet` (per-shard checkpoints)
- `data/textsec/processed/FinBert/finbert_master.parquet` (final output)

**Key Parameters**:
- `--manifest`: Path to shard manifest CSV (or use `--shards_dir`)
- `--device`: `cuda` or `cpu` (default: `cuda`)
- `--batch_size`: Batch size for FinBERT scoring (default: 64)
- `--max_length`: Max token length (default: 128)
- `--num_workers`: Parallel workers (default: min(8, cpu_count))
- `--fp16`: Use half-precision (reduces memory, faster on modern GPUs)
- `--scope`: `global`, `global_ex_us`, `country`, or `continent` (default: `global`)
- `--scope_value`: Country/continent code when using `country` or `continent` scope
- `--lag_days`: Minimum days after filing date to include (default: 7)
- `--checkpoint_dir`: Directory for intermediate checkpoints (default: `data/textsec/processed/FinBert/checkpoints`)
- `--checkpoint_every`: Write merged checkpoint every N shards (default: 5)

**USA Scope Special Handling**:
- When processing USA scope, creates separate aggregations:
  - Per-section: risk_factors, management_discussion_and_analysis
  - Total: aggregated across all sections
- Both stored in final output with `section` column

**SEC Sentiment Features Computed**:
- **Time-series features** (per gvkey, iid, section):
  - Momentum: `S_mom_1`, `S_mom_3`, `S_mom_6`
  - Percent change: `S_pctchg_1`
  - Moving averages: `S_ma_3`, `S_ma_6`, `S_ma_12`
  - Volatility: `S_vol_3`, `S_vol_6`, `S_vol_12`
  - EMA/MACD: `S_ema_6`, `S_ema_12`, `S_macd_12_26`, `S_macd_sig_9`
  - Z-score: `S_z_12`
  - Coefficient of variation: `S_cv_6`
- **Cross-sectional features** (per month_end, section):
  - Rank: `S_xs_rank`
  - Z-score: `S_xs_z`
- **Probability features**:
  - Entropy: `P_entropy`
  - Confidence: `prob_confidence`
- **Document features**:
  - Lag count: `doc_sent_count_lag1`
  - Intensity: `doc_sent_intensity_3`

---

### Step 3 Alternative: Multi-Instance Parallel Processing

**Script**: `run/run_multi_instance.py`

For systems with many CPU cores (e.g., Ryzen 9 7900X with 24 threads), you can split shards across multiple independent FinBERT processes for maximum parallelization. This is more efficient than using a single instance with many workers.

**Command**:
```bash
python run/run_multi_instance.py \
  --script run/sec_finbert_monthly_optimized.py \
  --manifest "data/textsec/interim/filings_clean_manifest.csv" \
  --instances 6 \
  --worker_per_instance 4 \
  --device cuda \
  --batch_size 64 \
  --fp16 \
  --final_out "data/textsec/processed/FinBert/finbert_master.parquet"
```

**What it does**:
- Discovers all shards from manifest or directory
- Splits shards evenly across N independent instances
- Launches each instance as a separate process with its own workers
- Each instance processes its assigned shards independently
- Automatically merges all results from all instances
- Handles duplicate detection and aggregation (weighted averages for sentiment metrics)
- Recomputes global SEC features on merged result

**Key Parameters**:
- `--script`: Path to FinBERT script (default: `run/sec_finbert_monthly_optimized.py`)
- `--instances`: Number of independent instances to run (default: 4)
- `--worker_per_instance`: Workers per instance (default: 4)
- `--manifest`: Path to shard manifest CSV (or use `--shards_dir`)
- `--device`: `cuda` or `cpu` (default: `cuda`)
- `--batch_size`: Batch size for FinBERT scoring (default: 64)
- `--max_length`: Max token length (default: 128)
- `--fp16`: Use half-precision (faster, less memory)
- `--final_out`: Final merged output file path (default: `data/textsec/processed/FinBert/finbert_master.parquet`)
- `--out_base`: Base directory for per-instance outputs (default: `data/textsec/processed/FinBert/instances`)
- `--limit_shards`: Limit number of shards to process (for testing)

**When to Use**:
- **Recommended for**: Systems with 16+ CPU cores and multiple GPUs (or single GPU with high VRAM)
- **Example**: 24-core CPU → use `--instances 6 --worker_per_instance 4` (24 total workers)
- **Benefits**: Better resource utilization, faster processing on high-end systems
- **Trade-off**: Uses more memory/GPU resources simultaneously

**Output Structure**:
- Per-instance outputs: `{out_base}/instance_{N}/result.parquet`
- Final merged output: `{final_out}`

**Merging Logic**:
- Automatically detects and handles duplicates across instances
- Uses document-count-weighted averages for sentiment probabilities
- Recomputes global SEC features (momentum, moving averages, etc.) on final merged data

**Example for 24-core system**:
```bash
python run/run_multi_instance.py \
  --manifest "data/textsec/interim/filings_clean_manifest.csv" \
  --instances 6 \
  --worker_per_instance 4 \
  --device cuda \
  --batch_size 64 \
  --fp16
```

This launches 6 independent processes, each with 4 workers (24 total workers), maximizing CPU/GPU utilization.

---

## Hardware Configuration

### CPU-Only Systems

**Recommended Settings**:

```bash
# Step 1: Feature Extraction
python run/make_sec_features.py --num_workers 8 --us_num_workers 2

# Step 2: Company Linking
python run/link_global_shards.py --num_workers 4 ...

# Step 3: Sentiment Analysis
python run/sec_finbert_monthly_optimized.py \
  --device cpu \
  --batch_size 32 \
  --num_workers 4
```

**Adjustments**:
- Reduce `--num_workers` if you have < 8 CPU cores
- Lower `--batch_size` for FinBERT if RAM < 16GB (try 16 or 8)
- Use `--us_num_workers 1` if experiencing memory issues

### GPU Systems (Recommended)

**NVIDIA GPU with 8GB+ VRAM**:

```bash
# Step 1: Feature Extraction (CPU-only, no change)
python run/make_sec_features.py --num_workers 16 --us_num_workers 2

# Step 2: Company Linking (CPU-only, no change)
python run/link_global_shards.py --num_workers 4 ...

# Step 3: Sentiment Analysis
python run/sec_finbert_monthly_optimized.py \
  --device cuda \
  --batch_size 64 \
  --num_workers 4 \
  --fp16
```

**GPU with 4-6GB VRAM**:
```bash
# Step 3: Reduce batch size
python run/sec_finbert_monthly_optimized.py \
  --device cuda \
  --batch_size 32 \
  --num_workers 4 \
  --fp16
```

**GPU with 2-4GB VRAM**:
```bash
# Step 3: Further reduce batch size
python run/sec_finbert_monthly_optimized.py \
  --device cuda \
  --batch_size 16 \
  --num_workers 2 \
  --fp16
```

### High-End Systems (32GB+ RAM, 16+ CPU cores, GPU)

**Maximum Performance**:

```bash
# Step 1: Use all cores
python run/make_sec_features.py --num_workers 16 --us_num_workers 4

# Step 2: More workers (but watch memory)
python run/link_global_shards.py --num_workers 8 ...

# Step 3: Larger batches (single instance)
python run/sec_finbert_monthly_optimized.py \
  --device cuda \
  --batch_size 128 \
  --num_workers 8 \
  --fp16
```

**Alternative: Multi-Instance (Recommended for 20+ cores)**:

For systems with many CPU cores (20+), use `run_multi_instance.py` for better parallelization:

```bash
# Step 3: Multi-instance parallel processing
python run/run_multi_instance.py \
  --manifest "data/textsec/interim/filings_clean_manifest.csv" \
  --instances 6 \
  --worker_per_instance 4 \
  --device cuda \
  --batch_size 64 \
  --fp16
```

This splits work across 6 independent processes (24 total workers), maximizing resource utilization.

### Low-Memory Systems (<16GB RAM)

**Conservative Settings**:

```bash
# Step 1: Fewer workers
python run/make_sec_features.py --num_workers 4 --us_num_workers 1

# Step 2: Sequential processing
python run/link_global_shards.py --num_workers 2 ...

# Step 3: Small batches, CPU mode
python run/sec_finbert_monthly_optimized.py \
  --device cpu \
  --batch_size 8 \
  --num_workers 2
```

### Multi-GPU Systems

**Current implementation uses single GPU per worker**. For multi-GPU:
- Reduce `--num_workers` to 1-2 per GPU
- Manually split work by scope (process different countries/regions separately)
- Or modify script to use `torch.nn.DataParallel` (not currently implemented)

---

## Script Details

### make_sec_features.py

**Purpose**: Normalize raw SEC filing data into standardized parquet shards.

**Key Features**:
- Supports multiple input formats (pkl, parquet, csv, json, jsonl)
- Handles US vs Global file formats differently
- Creates both fused (single text column) and split (rf/mgmt preserved) outputs for US
- Memory-efficient with reduced worker count for US files
- Automatic sequential retry for failed files
- Creates manifest tracking all output shards

**Column Normalization**:
- Maps various name columns → `company_name`
- Maps various date columns → `filing_date`
- Maps various form columns → `form_type`
- Creates standardized `country`, `continent`, `excntry` columns
- For US: handles `rf` (risk_factors) and `mgmt` (management_discussion) columns

### link_global_shards.py

**Purpose**: Link company names in filings to Compustat identifiers (gvkey, iid).

**Matching Strategy**:
1. **Exact matching tiers** (fast, high precision):
   - Normalizes company names (removes punctuation, stopwords, etc.)
   - Matches by company + country + date (exact or year)
   - Allows cross-country equivalence (e.g., HKG ↔ CHN)
   
2. **Fuzzy matching** (slower, for remaining rows):
   - Constrained by country and year buckets
   - Uses token-based similarity (SequenceMatcher)
   - Special thresholds per market (CAN: 0.90, HKG: 0.86, CHN: 0.88)
   - Performance-limited to prevent hangs

**Performance Safeguards**:
- Maximum 20,000 rows per shard for fuzzy matching
- Maximum 2,000-5,000 candidates per row
- Atomic writes prevent corruption on crashes
- Progress logging every 500-1000 rows

### sec_finbert_monthly_optimized.py

**Purpose**: Compute FinBERT sentiment scores and create monthly aggregated time series.

**Model**: Uses `yiyanghkust/finbert-tone` (3-class: negative, neutral, positive)

**Output Metrics**:
- `prob_neg_mean`, `prob_neu_mean`, `prob_pos_mean`: Probability distributions
- `sent_pos_minus_neg_mean`: Positive minus negative (main sentiment score)
- `doc_sent_count`: Number of documents aggregated

**Aggregation Level**:
- Monthly per (gvkey, iid, month_end, country, continent, section)
- USA scope includes both per-section and total aggregations

**Features Computed**:
- 20+ time-series features (momentum, moving averages, volatility, etc.)
- Cross-sectional features (rank, z-score within month)
- Document intensity features

### run_multi_instance.py

**Purpose**: Launch multiple independent FinBERT instances in parallel for maximum CPU/GPU utilization on high-end systems.

**Key Features**:
- Splits shards evenly across N independent processes
- Each instance runs with its own worker pool
- Automatically merges results from all instances
- Handles duplicate detection and weighted aggregation
- Recomputes global SEC features on merged data

**Use Cases**:
- Systems with 16+ CPU cores
- Multiple GPUs or single GPU with high VRAM
- Large datasets requiring maximum throughput

**Merging Strategy**:
- Detects duplicates on (gvkey, iid, month_end, country, continent, section)
- Uses document-count-weighted averages for sentiment probabilities
- Preserves identifiers and metadata
- Recomputes time-series features globally after merge

---

## Troubleshooting

### Common Issues

**1. BrokenProcessPool errors**

**Symptoms**: Workers crash during parallel processing

**Solutions**:
- Reduce `--num_workers` (already optimized in defaults)
- Files automatically retry sequentially
- Check available RAM/disk space
- For Step 2: reduce `--num_workers` to 2-4

**2. Out of Memory (OOM) errors**

**Symptoms**: CUDA OOM or system runs out of RAM

**Solutions**:
- **Step 1**: Already uses reduced workers for US files
- **Step 2**: Reduce `--num_workers` to 2
- **Step 3**: Reduce `--batch_size` (try 32, 16, or 8), use `--fp16`, or switch to `--device cpu`

**3. File corruption**

**Symptoms**: "Parquet magic bytes not found" errors

**Solutions**:
- Step 2 now uses atomic writes (temp file → rename) to prevent corruption
- If file is corrupted: regenerate from source by re-running Step 1
- Check disk space

**4. HK/Taiwan/China shards hang**

**Symptoms**: Very slow processing, seems to hang

**Solutions**:
- Already addressed in Step 2: limits fuzzy matching to 20K rows per shard
- Processing should complete in 10-30 minutes instead of hours
- If still slow: reduce `MAX_FUZZY_ROWS` in code (currently 20000)

**5. Missing gvkey/iid after linking**

**Possible reasons**:
- Company name not in merge table
- Date mismatch (filing date vs merge table date)
- Country mismatch
- Fuzzy matching threshold too strict

**Solutions**:
- Check name merge table coverage
- Review linking statistics in output
- Manually verify a few examples
- Consider adding more name merge files

**6. FinBERT slow on CPU**

**Solutions**:
- Use GPU if available (`--device cuda`)
- Reduce `--batch_size`
- Use `--fp16` even on CPU (if supported)
- Process smaller scopes (use `--scope country` with specific countries)

**7. Import errors in workers**

**Symptoms**: Import errors when using multiprocessing

**Solutions**:
- Ensure all dependencies installed in active conda environment
- Check Python version (3.11+ recommended)
- Verify CUDA version matches PyTorch installation

---

## Output Files

### Step 1 Outputs

**Fused Shards**: `data/textsec/interim/filings_clean_shards/{country}/{filename}.parquet`
- Schema: `company_name`, `filing_date`, `form_type`, `section`, `text`, `country`, `continent`, `excntry`, `gvkey`, `iid`, ...

**US Split Shards**: `data/textsec/interim/filings_clean_shards_us_split/USA/{filename}.parquet`
- Schema: Same as fused, but has `rf` and `mgmt` columns instead of `text`

**Manifest**: `data/textsec/interim/filings_clean_manifest.csv`
- Columns: `shard_path`, `country`, `flavor` (fused/split), `source_file`

### Step 2 Outputs

**Updated Shards**: Same location as Step 1, with `gvkey` and `iid` columns filled in

### Step 3 Outputs

**Checkpoints**: `data/textsec/processed/FinBert/checkpoints/{shard_name}__monthly.parquet`
- Per-shard monthly aggregations

**Master File**: `data/textsec/processed/FinBert/finbert_master.parquet`
- Schema:
  - Identifiers: `gvkey`, `iid`, `company_name`, `excntry`, `country`, `continent`, `section`
  - Time: `month_end`
  - Sentiment: `prob_neg_mean`, `prob_neu_mean`, `prob_pos_mean`, `sent_pos_minus_neg_mean`, `doc_sent_count`
  - Features: `S_mom_1`, `S_mom_3`, `S_ma_3`, `S_vol_3`, `S_xs_rank`, etc. (20+ features)

---

## Performance Benchmarks

**Typical Processing Times** (approximate, depends on data size):

| Step | System | Time |
|------|--------|------|
| Step 1 (21 US + 26 Global files) | 16-core CPU, 32GB RAM | 5-15 minutes |
| Step 2 (68 shards) | 4-core CPU, 16GB RAM | 10-30 minutes |
| Step 3 (68 shards) | RTX 3090, fp16, batch=64 | 30-60 minutes |
| Step 3 (68 shards) | CPU-only, batch=32 | 2-4 hours |

**Memory Usage**:
- Step 1: ~2-4GB per worker
- Step 2: ~4-8GB per worker (large merge tables)
- Step 3: ~2-4GB GPU VRAM (with batch_size=64), ~8-16GB system RAM

---

## Advanced Usage

### Processing Specific Countries

**Step 2**:
```bash
python run/link_global_shards.py \
  --countries "CAN,HKG,CHN" \
  ...
```

**Step 3**:
```bash
python run/sec_finbert_monthly_optimized.py \
  --scope country \
  --scope_value "CAN" \
  ...
```

### Processing Only Global (Exclude US)

**Step 3**:
```bash
python run/sec_finbert_monthly_optimized.py \
  --scope global_ex_us \
  ...
```

### Custom Checkpoint Directory

**Step 3**:
```bash
python run/sec_finbert_monthly_optimized.py \
  --checkpoint_dir "custom/path/checkpoints" \
  --checkpoint_every 10 \
  ...
```

### Debug Mode

**Step 1**:
```bash
python run/make_sec_features.py --debug ...
```

---

## Version History

- **2025-01**: Initial pipeline implementation
  - Optimized US file processing with reduced workers
  - Added atomic writes to prevent corruption
  - Added performance limits for fuzzy matching
  - Sequential retry for failed shards
  - USA scope handling in FinBERT

---

## License

[Add your license information here]

---

## Contact

[Add contact information here]

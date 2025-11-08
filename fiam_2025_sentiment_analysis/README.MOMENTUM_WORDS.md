# 🧭 Sector Word Momentum — Progressive Dictionary Pipeline

This pipeline builds **sector-level text momentum outputs** from SEC filings using a progressive (non-lookahead) dictionary. Words are mapped to sectors dynamically as they first appear in filings, ensuring no forward information leakage.

The pipeline processes **~1M filings** (13.5GB input) into sector-level momentum signals, with memory-optimized processing at each stage.

---

## 📋 Pipeline Overview

The complete pipeline consists of **7 main steps**:

1. **Combine Parquet Shards** → Merge distributed shards into a single `filings_clean.parquet`
2. **Build Monthly Word Trends** → Generate word-level TF/DF statistics and momentum features
3. **Freeze Sector Dictionary** → Validate pre-2015 words to prevent lookahead bias
4. **Build Static Sector Momentum** → Create baseline sector signals using frozen dictionary only
5. **Map New Words to Sectors** → Discover and map words that appeared after 2015
6. **Validate and Activate** → Filter new words by quality thresholds
7. **Build Progressive Momentum** → Combine frozen + new words into final progressive outputs

---

## 📦 Final Outputs

### 1. `sector_word_frequency_monthly_progressive.parquet`
**Description:**  
Monthly time series of sector-level text frequency (`tf_share`) using the **progressive vocabulary**.  
Includes both original "frozen" dictionary words (pre-2015) and new words mapped after 2015.

**Key columns:**
| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Month end |
| `sector` | string | One of the 12 sectors |
| `sector_tf_share` | float | Total normalized frequency for all words mapped to that sector in the month |

**Usage:**  
- Base layer for feature creation  
- Combine with sector returns, macro data, or sentiment indices  
- Plot normalized intensity per sector through time

---

### 2. `sector_word_momentum_monthly_progressive.parquet`
**Description:**  
Same as above, but includes **momentum features** (`mom_3m`, `mom_6m`, `mom_12m`) calculated using rolling percentage changes in `tf_share`.

**Key columns:**
| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Month end |
| `sector` | string | Sector name |
| `sector_tf_share` | float | Monthly text frequency |
| `mom_3m` | float | 3-month momentum |
| `mom_6m` | float | 6-month momentum |
| `mom_12m` | float | 12-month momentum |

**Usage:**  
- Direct input for predictive models or cross-sectional regressions  
- Compare to frozen baseline (`StaticDict/sector_word_momentum_monthly.parquet`) to see added informational value of new language

---

## 📊 Understanding `sector_tf_share` and Momentum Calculation

### What is `sector_tf_share`?

The `sector_tf_share` column represents **the sector's share of total tracked word frequency** for that month. It is calculated as follows:

1. **Word-level aggregation**: Each word in the sector dictionary has a `tf_share` value (term frequency share) representing its proportion of total word frequency in that month.

2. **Sector mapping**: Words are mapped to sectors using either:
   - **Frozen dictionary**: Pre-2015 words validated to exist before the cutoff
   - **Activated words**: New words mapped to sectors after 2015

3. **Sector aggregation**: For each sector-month combination, `sector_tf_share` is the **sum of all `tf_share` values** for words mapped to that sector.

4. **Multi-sector words**: Words assigned to multiple sectors have their `tf_share` **split equally** across all assigned sectors.

### Key Characteristics

- **Typical values**: Range from ~0.0005 to ~0.05 (small because only dictionary words are included)
- **Sum per month**: The sum of `sector_tf_share` across all 12 sectors per month is typically ~0.03 (not 1.0), indicating that only a subset of all words in filings are tracked via the sector dictionaries
- **Interpretation**: Higher values indicate that sector-related vocabulary is more prominent in SEC filings that month

### How Momentum is Calculated

Momentum features (`mom_1m`, `mom_3m`, `mom_6m`, `mom_12m`) measure the **percentage change** in `sector_tf_share` over rolling windows:

**Formula:**
```
mom_k = (current_sector_tf_share - sector_tf_share_k_months_ago) / sector_tf_share_k_months_ago
```

**Example:**
- If `sector_tf_share` for Technology sector is:
  - January 2024: 0.0010
  - July 2024: 0.0015
- Then `mom_6m` for July 2024 = (0.0015 - 0.0010) / 0.0010 = **0.50 (50% increase)**

**Implementation details:**
- Uses a **safe percentage change function** that avoids division by very small numbers (threshold: 1e-9)
- When the previous value is too small (< 1e-9), momentum is set to `NaN` to prevent numerical instability
- Calculated separately for each sector using grouped time series

**Interpretation:**
- **Positive momentum**: Sector vocabulary is becoming more prominent in filings (increasing share)
- **Negative momentum**: Sector vocabulary is becoming less prominent (decreasing share)
- **High momentum**: Rapid change in sector-related language usage, potentially indicating shifting market focus or sector trends

---

## 🚀 Complete Reproduction Pipeline

### Step 1: Combine Parquet Shards

**Purpose:** Merge all processed parquet shards (generated by `make_sec_features.py` and optionally linked by `link_global_shards.py`) into a single consolidated file.

**Script:** `run/combine_shards_to_filings_clean.py`

**What it does:**
- Reads manifest CSV listing all shard paths
- Streams shards directly to output parquet file (memory-efficient)
- Enforces consistent schema (ensures ID columns like `gvkey` are strings)
- Outputs: `data/textsec/interim/filings_clean.parquet`

**Prerequisites:**
- `make_sec_features.py` must have run (generates shards + manifest)
- `link_global_shards.py` should have run if using global shards (adds `gvkey`, `iid`)

**Command:**
```bash
python run/combine_shards_to_filings_clean.py \
  --paths config/paths.yaml \
  --flavor fused \
  --output data/textsec/interim/filings_clean.parquet
```

**Expected output:**
- `filings_clean.parquet`: ~13.5GB, ~1M rows, columns: `text`, `date`, `gvkey`, `iid`, etc.

---

### Step 2: Build Monthly Word Trends

**Purpose:** Construct a monthly time series of word frequencies (TF/DF shares) from cleaned SEC filings, using a dynamically evolving lexicon.

**Script:** `src/textsec/build_word_trends.py`

**What it does:**
- Reads `filings_clean.parquet` (optimized: only `text` and `date` columns)
- Builds baseline lexicon on 2005–2010 filings
- Extends lexicon yearly using only past data (no forward-looking info)
- Generates monthly panel with word-level statistics:
  - `gram`, `date`, `tf`, `df_docs`, `month_tokens`, `month_docs`
  - `tf_share`, `df_share` (normalized frequencies)
  - Word-level momentum: `mom_1m`, `mom_3m`, `mom_6m`, `mom_12m`, `r3m`, `r6m`, `r12m`
- Uses incremental processing to handle large datasets (212M+ rows)

**Outputs:**
- `data/textsec/processed/word_trends_monthly.parquet` (~9.8GB, 212M rows)
- Optional CSV summaries in `summaries/` directory

**Command:**
```bash
python src/textsec/build_word_trends.py --paths config/paths.yaml
```

**Parallel processing (recommended for multi-core systems):**
```bash
python src/textsec/build_word_trends.py --paths config/paths.yaml --parallel --n_jobs 8
```

The `--parallel` flag processes all months within each year in parallel, since the lexicon for a given year is fixed before processing any months. This can significantly speed up processing on multi-core systems:
- **Without parallel**: Processes months sequentially (slower but uses less memory)
- **With parallel**: Processes 12 months per year in parallel (faster, uses more CPU cores)

**Parameters:**
- `--parallel`: Enable parallel month processing
- `--n_jobs N`: Number of parallel workers (default: auto-detect, uses CPU count - 1)

**Memory optimizations:**
- Column-selective reading (only `text`, `date` from input)
- Incremental panel building (writes quarterly chunks to disk)
- Two-pass momentum computation (month totals → shares/momentum)
- Polars lazy evaluation for streaming writes
- Parallel processing respects timeline (lexicon snapshots per year, months within year processed in parallel)

---

### Step 3: Freeze Sector Dictionary

**Purpose:** Create a leak-free dictionary by validating words against historical data. Only words that appeared before 2015 are kept, preventing lookahead bias.

**Script:** `run/freeze_sector_dictionary.py`

**What it does:**
- Starts with seed dictionary (`resources/sector_dictionary_seed.json`)
- Validates each word against `word_trends_monthly.parquet`:
  - Must first appear **≤ 2010-12-31** (`strict_year`)
  - Must appear in **≥ 3 distinct months** before 2015 (`min_pre_cutoff_months`)
  - Must appear in **≥ 2 months with docs > 0** (`min_pre_cutoff_doc_months`)
- Outputs frozen dictionary with only validated words

**Outputs:**
- `data/textsec/processed/FreezeDict/sector_dictionary.json` (frozen dictionary)
- `data/textsec/processed/FreezeDict/sector_word_appearance_report.csv` (audit report)

**Command:**
```bash
python run/freeze_sector_dictionary.py \
  --panel data/textsec/processed/word_trends_monthly.parquet \
  --sector_dict_in resources/sector_dictionary_seed.json \
  --sector_dict_out data/textsec/processed/FreezeDict/sector_dictionary.json \
  --appearance_report_out data/textsec/processed/FreezeDict/sector_word_appearance_report.csv \
  --cutoff_year 2015 \
  --strict_year 2010 \
  --min_pre_cutoff_months 3 \
  --min_pre_cutoff_doc_months 2
```

**Memory optimizations:**
- Column-selective reading (only `date`, `gram`, `tf_share`, `df_share`, `month_docs`)
- Polars support for large files (>5GB)

---

### Step 4: Build Static Sector Momentum

**Purpose:** Create baseline sector-level momentum signals using **only** the frozen dictionary (pre-2015 words). This serves as a baseline and is also needed as input for mapping new words.

**Script:** `run/make_sector_momentum.py`

**What it does:**
- Loads frozen dictionary
- Maps each word to its assigned sector
- Aggregates `tf_share` per sector per month
- Computes rolling momentum features: `mom_3m`, `mom_6m`, `mom_12m`
- Outputs baseline sector signals

**Outputs:**
- `data/textsec/processed/StaticDict/sector_word_frequency_monthly.parquet` (base frequencies)
- `data/textsec/processed/StaticDict/sector_word_momentum_monthly.parquet` (momentum signals)
- `data/textsec/processed/StaticDict/sector_momentum_summary.csv` (top-3 sectors per month)

**Command:**
```bash
python run/make_sector_momentum.py \
  --panel data/textsec/processed/word_trends_monthly.parquet \
  --sector_dict_in data/textsec/processed/FreezeDict/sector_dictionary.json \
  --sector_dict_out data/textsec/processed/FreezeDict/sector_dictionary.json \
  --first_trading_year 2015 \
  --use_share tf_share \
  --out_dir data/textsec/processed/StaticDict
```

**Note:** This step is required even for the progressive pipeline, as `sector_word_frequency_monthly.parquet` is needed as input for mapping new words.

---

### Step 5: Map New Words to Sectors

**Purpose:** Discover words that appeared after 2015 (not in frozen dictionary) and map them to sectors based on correlation with existing sector frequencies.

**Script:** `run/map_new_words_to_sectors.py`

**What it does:**
- Filters words to those that first appear after 2015 (`--restrict_to_oos`)
- For each new word, analyzes its first 12 months of usage
- Computes correlation with each sector's frequency series
- Assigns word to sector with highest correlation
- Outputs mapping file with sector assignments and quality metrics

**Outputs:**
- `data/textsec/processed/ProgressiveDict/new_words_sector_mapping_fixed12m.csv`

**Command:**
```bash
python run/map_new_words_to_sectors.py \
  --panel data/textsec/processed/word_trends_monthly.parquet \
  --sector_freq data/textsec/processed/StaticDict/sector_word_frequency_monthly.parquet \
  --frozen_dict data/textsec/processed/FreezeDict/sector_dictionary.json \
  --restrict_to_oos \
  --n_jobs 12 \
  --min_points 9 \
  --checkpoint_every 50000 \
  --out_dir data/textsec/processed/ProgressiveDict
```

**Parameters:**
- `--restrict_to_oos`: Only process words that first appear after 2015 (out-of-sample)
- `--min_points 9`: Require at least 9 months of data for correlation
- `--checkpoint_every 50000`: Save progress every 50k words (allows resume if interrupted)

---

### Step 6: Validate and Activate

**Purpose:** Filter new words by quality thresholds and ensure they meet activation criteria (no lookahead, sufficient correlation, etc.).

**Script:** `run/validate_activation_map.py`

**What it does:**
- Validates each mapped word against quality thresholds:
  - `p_min` (default 0.30): Minimum correlation with assigned sector
  - `m_min` (default 0.05): Minimum mean frequency
- Enforces cutoff year (words must appear only after 2015)
- Filters ambiguous words (assigned to multiple sectors)
- Outputs activation map with validated words and recommendations

**Outputs:**
- `data/textsec/processed/ProgressiveDict/new_words_sector_mapping_active_ALL.csv` (validated activations)
- `data/textsec/processed/ProgressiveDict/new_words_sector_mapping_recommendations.csv` (filtered words + reasons)

**Command:**
```bash
python run/validate_activation_map.py \
  --inp data/textsec/processed/ProgressiveDict/new_words_sector_mapping_fixed12m.csv \
  --out data/textsec/processed/ProgressiveDict/new_words_sector_mapping_active_ALL.csv \
  --reco_out data/textsec/processed/ProgressiveDict/new_words_sector_mapping_recommendations.csv \
  --p_min 0.30 \
  --m_min 0.05 \
  --cutoff_year 2015 \
  --enforce_cutoff
```

**Typical results:**
- ~165k candidate words
- ~121k words activated (73% pass rate)
- ~40k ambiguous words (filtered out)

---

### Step 7: Build Progressive Momentum (Final Outputs)

**Purpose:** Combine frozen dictionary (pre-2015 words) with validated new words to create the final progressive momentum outputs.

**Script:** `run/make_sector_momentum_progressive.py`

**What it does:**
- Loads frozen dictionary (344 words)
- Loads activation map (121k+ validated new words)
- Filters to `suggest_keep=True` words
- Maps words to sectors progressively (new words activate at their first appearance date)
- Aggregates sector frequencies including both frozen and new words
- Computes momentum features (`mom_3m`, `mom_6m`, `mom_12m`)
- Outputs final progressive momentum signals

**Outputs:**
- `data/textsec/processed/ProgressiveDict/sector_word_frequency_monthly_progressive.parquet`
- `data/textsec/processed/ProgressiveDict/sector_word_momentum_monthly_progressive.parquet`
- `data/textsec/processed/ProgressiveDict/sector_momentum_summary_progressive.csv`

**Command:**
```bash
python run/make_sector_momentum_progressive.py \
  --panel data/textsec/processed/word_trends_monthly.parquet \
  --frozen_dict data/textsec/processed/FreezeDict/sector_dictionary.json \
  --activations data/textsec/processed/ProgressiveDict/new_words_sector_mapping_active_ALL.csv \
  --use_share tf_share \
  --out_dir data/textsec/processed/ProgressiveDict \
  --filter_suggest_keep
```

---

## 📊 Pipeline Statistics

**Data volumes:**
- Input filings: ~1,069,082 rows, 13.5GB
- Word trends panel: ~212,437,612 rows, 9.8GB
- Unique words: 3,721,200 total grams

**Dictionary breakdown:**
- Frozen dictionary: 344 words (pre-2015)
- New word candidates: 165,328 words
- Validated & activated: 121,538 words (73.5% pass rate)
- **Total progressive vocabulary: 121,882 words** (344 frozen + 121,538 new)

**Sectors (12 total):**
- Commodities, Consumer_Discretionary, Consumer_Staples, Developed_ex-US, Emerging_Markets, Energy, Financials, Healthcare, Industrials, Materials, Technology, Utilities

---

## 🧠 Input Dependencies

The progressive momentum outputs depend on:
- `word_trends_monthly.parquet` → raw word statistics (TF, DF, momentum)
- `sector_dictionary.json` → frozen (pre-2015) seed dictionary
- `new_words_sector_mapping_active_ALL.csv` → validated new words + activation dates
- `StaticDict/sector_word_frequency_monthly.parquet` → baseline sector frequencies (for mapping new words)

All new words are validated to appear **only after 2015**, ensuring no lookahead.

---

## 🔧 Troubleshooting

### Memory Issues
- The pipeline uses memory-optimized processing at each step
- If you encounter "Killed" errors:
  - Ensure sufficient RAM (recommended: 32GB+)
  - Check disk space (intermediate files require ~50GB+)
  - Scripts automatically use column-selective reading and incremental processing

### Resume Functionality
- `build_word_trends.py`: Checks for `temp_panel_chunks/panel_base.parquet` to resume from panel consolidation
- `map_new_words_to_sectors.py`: Uses checkpointing (`--checkpoint_every`) to allow resume

### Schema Errors
- If you see ID column conversion errors (e.g., `gvkey`), ensure `combine_shards_to_filings_clean.py` was run to enforce consistent string types

---

## 💻 Hardware Adaptation Guide

If you're running the pipeline on a different machine with different hardware (different RAM, CPU cores, GPU), adjust these settings:

### RAM Adjustments

**Low RAM (< 16GB):**
- All scripts use column-selective reading and incremental processing by default
- Monitor disk space as intermediate files are used more heavily
- Consider reducing batch sizes if you encounter OOM errors
- For `build_word_trends.py`: The script already handles large files efficiently, but if issues persist, you may need to manually split the input file

**Medium RAM (16-32GB):**
- Default settings should work well
- Scripts automatically detect large files and use optimized paths

**High RAM (32GB+):**
- Default settings work optimally
- Can process larger chunks if desired

**Check available RAM:**
```bash
free -h  # Linux
# or
top -l 1 | grep "PhysMem"  # macOS
```

---

### CPU Core Adjustments

**Step 5 (map_new_words_to_sectors.py):**
- Adjust `--n_jobs` parameter based on your CPU cores
- **Formula**: `n_jobs = min(available_cores, 12)` (recommended max: 12)
- **Low-end CPU (2-4 cores)**: `--n_jobs 2` or `--n_jobs 4`
- **Mid-range CPU (6-8 cores)**: `--n_jobs 6` or `--n_jobs 8`
- **High-end CPU (12+ cores)**: `--n_jobs 12` (or higher if you have many cores)

**Check CPU cores:**
```bash
nproc  # Linux - number of cores
# or
sysctl -n hw.ncpu  # macOS - number of cores
```

**Example for 8-core CPU:**
```bash
python run/map_new_words_to_sectors.py \
  --panel data/textsec/processed/word_trends_monthly.parquet \
  --sector_freq data/textsec/processed/StaticDict/sector_word_frequency_monthly.parquet \
  --frozen_dict data/textsec/processed/FreezeDict/sector_dictionary.json \
  --restrict_to_oos \
  --n_jobs 8 \  # Adjusted for 8 cores
  --min_points 9 \
  --checkpoint_every 50000 \
  --out_dir data/textsec/processed/ProgressiveDict
```

---

### GPU Considerations

**Current Pipeline:** The pipeline does **not** use GPU acceleration by default. All processing is CPU-based.

**If you want to add GPU support:**
- `map_new_words_to_sectors.py`: Could potentially benefit from GPU for correlation calculations (would require custom implementation)
- `build_word_trends.py`: Text processing is primarily CPU-bound
- Most steps rely on pandas/polars/pyarrow, which are CPU-optimized

**For GPU-enabled text processing:**
- Consider using CuPy or RAPIDS cuDF for GPU-accelerated DataFrame operations (requires NVIDIA GPU with CUDA)
- This would require significant code modifications

---

### Disk Space Requirements

**Total disk space needed:**
- Input data: ~13.5GB (`filings_clean.parquet`)
- Intermediate files: ~50GB+ (including temporary chunks)
- Final outputs: ~100MB (final progressive momentum files)

**Minimum recommended:** 100GB free space

**Check disk space:**
```bash
df -h  # Linux/macOS - shows disk usage
```

**Cleanup intermediate files (after successful completion):**
```bash
# Remove temporary chunk files from build_word_trends.py
rm -rf data/textsec/processed/temp_panel_chunks/
```

---

### Performance Tuning Checklist

**Before running on new hardware:**

1. **Check system resources:**
   ```bash
   # RAM
   free -h
   
   # CPU cores
   nproc
   
   # Disk space
   df -h
   ```

2. **Adjust parallelization:**
   - Set `--n_jobs` in Step 5 to match CPU cores
   - Leave other steps at defaults (they're already optimized)

3. **Monitor during execution:**
   ```bash
   # Watch memory usage
   watch -n 1 free -h
   
   # Or use htop/top
   htop
   ```

4. **If memory issues persist:**
   - Reduce `--n_jobs` in Step 5
   - Ensure sufficient swap space is configured
   - Process smaller batches by modifying `--checkpoint_every`

---

### Recommended Hardware Configurations

**Minimum viable:**
- RAM: 16GB
- CPU: 4 cores
- Disk: 100GB free
- Settings: `--n_jobs 4` in Step 5

**Recommended:**
- RAM: 32GB+
- CPU: 8-12 cores
- Disk: 200GB+ free
- Settings: `--n_jobs 8-12` in Step 5

**Optimal:**
- RAM: 64GB+
- CPU: 16+ cores
- Disk: 500GB+ free
- Settings: `--n_jobs 12` in Step 5

---

## 📚 Related Documentation

- `README_SECTOR.md` → Static dictionary pipeline (frozen words only, no evolution)
- `config/paths.yaml` → Paths configuration for all inputs/outputs

---

## 🎯 Key Design Principles

1. **No Lookahead**: All validation ensures words only use information available up to their activation date
2. **Progressive Vocabulary**: Dictionary evolves over time while maintaining leak-free guarantees
3. **Memory Efficiency**: Large datasets (13.5GB input, 9.8GB intermediate) processed incrementally
4. **Quality Control**: Multi-stage validation (freeze → map → validate → activate)

#!/usr/bin/env python3
"""
Add Decay Feature (EMA) to Word Trends - GPU Optimized
======================================================
Purpose:
    Adds exponential moving average (EMA) decay to word frequency trends,
    creating a smoothed `tf_share_decay` column alongside the raw `tf_share`.
    
    The decay feature gives each word a memory across time by smoothing its
    frequency with an exponential moving average. Older mentions fade but still
    influence the trend, making sector-level momentum more stable.

    **GPU-OPTIMIZED**: Uses RAPIDS cuDF and Dask-cuDF for multi-GPU acceleration
    on A100 GPUs. Processes data in parallel across multiple GPUs.

Inputs:
    --panel : Path to word trends parquet (`word_trends_monthly.parquet`)
    --alpha : Decay rate (default: 0.3)
              Higher α → faster reaction, less memory
              Lower α → slower reaction, longer memory
    --use_gpu : Use GPU acceleration (default: True)
    --n_gpus : Number of GPUs to use (default: 2 for A100 setup)
    --chunk_size : Words per chunk for GPU processing (default: 100000)

Outputs:
    - Updates `word_trends_monthly.parquet` with new `tf_share_decay` column

Core Logic:
    1. Loads word trends panel (GPU-accelerated)
    2. For each word (gram), computes EMA decay over time:
       decayed_value_t = α × current_value_t + (1-α) × decayed_value_t-1
    3. Adds `tf_share_decay` column
    4. Saves updated panel

Mechanism:
    At each month t:
        decayed_value_t = α × current_value_t + (1-α) × decayed_value_t-1
    
    Uses only the previous decayed value — making it optimized (O(1) per step).
    GPU-accelerated using cuDF operations for maximum performance.

Command:
    python add_decay_feature.py --panel data/textsec/processed/word_trends_monthly.parquet --alpha 0.3 --n_gpus 2

Dependencies:
    For GPU acceleration (recommended for A100):
        - cudf (RAPIDS cuDF)
        - dask-cudf (Dask-cuDF for multi-GPU)
        - dask-cuda (Dask CUDA cluster)
    
    For CPU fallback:
        - pandas (auto-detected if GPU libraries unavailable)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import os
import logging
from datetime import datetime

# Try to import GPU libraries, fallback to CPU
try:
    import cudf
    import dask_cudf
    from dask.distributed import Client
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    import pandas as pd
    print("[WARN] GPU libraries (cudf, dask_cudf) not available. Falling back to CPU (pandas).")

# Fallback pandas import
import pandas as pd


def compute_ema_decay_gpu(series, alpha: float):
    """
    Compute exponential moving average decay for a time series on GPU.
    
    Formula: decayed_value_t = α × current_value_t + (1-α) × decayed_value_t-1
    
    GPU-optimized using cuDF operations.
    
    Args:
        series: cuDF Series with time series values (should be sorted by date)
        alpha: Decay rate (0 < alpha <= 1)
    
    Returns:
        cuDF Series with decayed values
    """
    if len(series) == 0:
        return series
    
    # Handle NaN values
    nan_mask = series.isna()
    
    # Forward-fill NaN values so decay computation continues smoothly
    series_filled = series.ffill().fillna(0.0)
    
    # Compute EMA using cuDF ewm (exponential weighted moving average)
    # adjust=False gives the pure recursive formula: y_t = α*x_t + (1-α)*y_{t-1}
    decayed = series_filled.ewm(alpha=alpha, adjust=False).mean()
    
    # Where original had NaN, set decayed[i] = decayed[i-1] (no update when data missing)
    if nan_mask.any():
        decayed_ffilled = decayed.ffill().fillna(0.0)
        decayed = decayed.where(~nan_mask, decayed_ffilled)
    
    return decayed


def compute_ema_decay_cpu(series: pd.Series, alpha: float) -> pd.Series:
    """
    Compute exponential moving average decay for a time series on CPU.
    
    Formula: decayed_value_t = α × current_value_t + (1-α) × decayed_value_t-1
    
    Uses pandas' ewm for computation.
    
    Args:
        series: pandas Series with time series values (should be sorted by date)
        alpha: Decay rate (0 < alpha <= 1)
    
    Returns:
        pandas Series with decayed values
    """
    if len(series) == 0:
        return series
    
    nan_mask = series.isna()
    series_filled = series.ffill().fillna(0.0)
    decayed = series_filled.ewm(alpha=alpha, adjust=False).mean()
    
    if nan_mask.any():
        decayed_ffilled = decayed.ffill().fillna(0.0)
        decayed = decayed.where(~nan_mask, decayed_ffilled)
    
    return decayed


def add_decay_feature_gpu(df_gpu, alpha: float = 0.3, n_gpus: int = 2):
    """
    Add tf_share_decay column using GPU acceleration with multi-GPU support.
    
    FIXED: Uses partition-by-gram strategy to co-locate same words, then processes
    each partition by converting to pandas for stable grouped EMA, then back to cuDF.
    This avoids cuDF groupby iteration which is not supported.
    
    Args:
        df_gpu: dask_cudf DataFrame with columns (gram, date, tf_share, ...)
        alpha: Decay rate (default: 0.3)
        n_gpus: Number of GPUs to use
    
    Returns:
        dask_cudf DataFrame with added tf_share_decay column
    """
    import logging
    import time
    logger = logging.getLogger(__name__)
    
    logger.info(f"Computing EMA decay with α={alpha} on {n_gpus} GPUs...")
    logger.info(f"Input: {df_gpu.npartitions} partitions")
    
    # Ensure proper types
    logger.info("Ensuring proper data types...")
    df_gpu["date"] = df_gpu["date"].astype("datetime64[ns]")
    logger.info("Data types completed")
    
    # Check that tf_share exists
    if "tf_share" not in df_gpu.columns:
        logger.error("Panel missing 'tf_share' column")
        raise ValueError("Panel must contain 'tf_share' column")
    
    # CRITICAL FIX: Partition by gram first to co-locate same words
    # This avoids global shuffle and ensures all rows for a word are in same partition
    logger.info("Partitioning by 'gram' to co-locate same words (avoiding global sort)...")
    # Check if gram is already index
    current_index = df_gpu.index.name if hasattr(df_gpu.index, 'name') else None
    if current_index != "gram":
        # Set gram as index to co-locate words (shuffle="tasks" uses hash partitioning)
        df_gpu = df_gpu.set_index("gram", shuffle="tasks")
        logger.info("Set 'gram' as index for co-location")
    else:
        logger.info("'gram' already set as index")
    
    # Repartition for better load balancing (but words stay co-located within partitions)
    logger.info("Repartitioning for better load balancing...")
    old_partitions = df_gpu.npartitions
    repart_start = time.time()
    df_gpu = df_gpu.repartition(partition_size="256MB")
    repart_elapsed = time.time() - repart_start
    logger.info(f"Repartitioned from {old_partitions} to {df_gpu.npartitions} partitions in {repart_elapsed:.2f}s")
    logger.info(f"  Average partition size: ~{256}MB target")
    
    # Use map_partitions for better GPU utilization
    # This processes each partition independently on its assigned GPU
    def process_partition(partition):
        """
        Process a partition: convert to pandas for stable grouped EMA, then back to cuDF.
        This avoids cuDF groupby iteration which is not supported.
        """
        import cudf
        import pandas as pd
        import logging
        import time
        logger = logging.getLogger(__name__)
        
        # Convert to pandas if needed (if partition is cuDF, convert to pandas)
        partition_start = time.time()
        partition_rows_initial = len(partition)
        partition_words_initial = partition.index.nunique() if hasattr(partition.index, 'nunique') else partition['gram'].nunique() if 'gram' in partition.columns else 0
        
        logger.info(f"  [PARTITION] Starting processing: {partition_rows_initial:,} rows, ~{partition_words_initial:,} words")
        
        if isinstance(partition, cudf.DataFrame):
            convert_start = time.time()
            pdf = partition.to_pandas()
            logger.info(f"  [PARTITION] Converted cuDF→pandas in {time.time() - convert_start:.2f}s")
        else:
            pdf = partition
        
        # Reset index to get gram back as column for groupby (gram is currently index)
        pdf = pdf.reset_index()
        
        # Sort within partition by gram and date
        sort_start = time.time()
        pdf = pdf.sort_values(["gram", "date"], kind="mergesort")
        logger.info(f"  [PARTITION] Sorted by gram+date in {time.time() - sort_start:.2f}s")
        
        # Compute EMA decay using pandas groupby (stable and supported)
        # This is the critical fix: use pandas for grouped EMA, not cuDF iteration
        # CRITICAL: Ensure no forward-looking bias - data is sorted by date, transform preserves order
        def compute_ema_pandas_series(s):
            """Compute EMA on a pandas Series. 
            Series is already in chronological order (DataFrame was sorted by date before grouping).
            The transform() function preserves the original row order, so values are chronological.
            """
            # DataFrame was sorted by ["gram", "date"] before grouping, so Series is already chronological
            # No need to sort - transform() preserves order from original DataFrame
            s_filled = s.ffill().fillna(0.0)
            return s_filled.ewm(alpha=alpha, adjust=False).mean()
        
        ema_start = time.time()
        unique_words = pdf['gram'].nunique()
        logger.info(f"  [PARTITION] Computing EMA for {unique_words:,} unique words...")
        
        # CRITICAL: transform() preserves the order of rows from the original DataFrame
        # Since pdf was sorted by ["gram", "date"], values within each group are chronological
        pdf["tf_share_decay"] = (
            pdf.groupby("gram", sort=False)["tf_share"]
              .transform(compute_ema_pandas_series)
        )
        
        pdf["tf_share_decay"] = pdf["tf_share_decay"].fillna(0.0)
        logger.info(f"  [PARTITION] EMA computed in {time.time() - ema_start:.2f}s")
        
        # Convert back to cuDF
        convert_back_start = time.time()
        result = cudf.from_pandas(pdf)
        logger.info(f"  [PARTITION] Converted pandas→cuDF in {time.time() - convert_back_start:.2f}s")
        
        # Set gram back as index to match original structure
        if "gram" in result.columns:
            result = result.set_index("gram")
        
        partition_elapsed = time.time() - partition_start
        logger.info(f"  [PARTITION] ✓ Complete: {len(result):,} rows in {partition_elapsed:.2f}s ({partition_rows_initial/partition_elapsed:,.0f} rows/s)")
        
        return result
    
    # Apply processing to each partition
    logger.info("Applying EMA decay to partitions (GPU-parallel, CPU-EMA per partition)...")
    df_gpu = df_gpu.map_partitions(
        process_partition,
        meta={
            **{col: df_gpu[col].dtype for col in df_gpu.columns},
            "tf_share_decay": "float64"
        }
    )
    
    # Reset index to get gram back as column (if it was set as index)
    logger.info("Resetting index to restore 'gram' as column...")
    df_gpu = df_gpu.reset_index()
    
    return df_gpu


def add_decay_feature_cpu(panel: pd.DataFrame, alpha: float = 0.3) -> pd.DataFrame:
    """
    Add tf_share_decay column to the panel using EMA decay per word (CPU version).
    
    Args:
        panel: pandas DataFrame with columns (gram, date, tf_share, ...)
        alpha: Decay rate (default: 0.3)
    
    Returns:
        pandas DataFrame with added tf_share_decay column
    """
    df = panel.copy()
    
    # Ensure proper types/order
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["gram", "date"], kind="mergesort").reset_index(drop=True)
    
    # Check that tf_share exists
    if "tf_share" not in df.columns:
        raise ValueError("Panel must contain 'tf_share' column")
    
    # Group by word and apply EMA decay
    print(f"[DECAY] Computing EMA decay with α={alpha} per word (CPU)...")
    g = df.groupby("gram", sort=False)
    
    # Use transform to keep index aligned
    # CRITICAL: Ensure no forward-looking bias - data is sorted by date, transform preserves order
    # The transform() function preserves the order of rows from the original DataFrame
    # Since df was sorted by ["gram", "date"], values within each group are chronological
    df["tf_share_decay"] = g["tf_share"].transform(
        lambda s: compute_ema_decay_cpu(s, alpha)
    )
    
    # Fill any remaining NaNs with 0 (for words that had all NaN tf_share)
    df["tf_share_decay"] = df["tf_share_decay"].fillna(0.0)
    
    return df


def setup_dask_cluster(n_gpus: int = 2):
    """Setup Dask cluster for multi-GPU processing."""
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    import os
    
    print(f"[DASK] Setting up Dask cluster with {n_gpus} GPUs...")
    
    # Disable UCX to avoid protocol issues - use TCP instead
    os.environ["DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT"] = "False"
    
    # CRITICAL FIX: Use conservative memory limits for 40GB cards (many Lightning A100s are 40GB, not 80GB)
    # Let Dask handle defaults or use safe conservative values
    cluster = LocalCUDACluster(
        n_workers=n_gpus,
        threads_per_worker=1,
        # Leave memory_limit as default (Dask will auto-detect) or use conservative value
        # device_memory_limit="28GB",  # Safe for 40GB cards (leave commented to use defaults)
        # rmm_pool_size="20GB",  # Conservative RMM pool (leave commented to use defaults)
        enable_nvlink=False,  # Disable NVLink to avoid protocol issues
        jit_unspill=True,  # Enable JIT unspill for better memory management
        protocol="tcp",  # Use TCP instead of UCX to avoid protocol issues
        interface="lo",  # Use loopback interface
        enable_tcp_over_ucx=False,  # Disable UCX completely
    )
    
    try:
        client = Client(cluster)
        print(f"[DASK] Cluster ready: {client}")
        print(f"[DASK] Dashboard: {client.dashboard_link}")
        return client, cluster
    except Exception as e:
        print(f"[ERROR] Failed to create client: {e}")
        cluster.close()
        raise


def setup_logging(log_file: Path = None):
    """Setup logging to both console and file."""
    # Create log directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Default log file name with timestamp
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"add_decay_feature_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger, log_file


def main():
    parser = argparse.ArgumentParser(
        description="Add EMA decay feature to word trends panel (GPU-optimized)"
    )
    parser.add_argument(
        "--panel",
        required=True,
        type=Path,
        help="Path to word_trends_monthly.parquet"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Decay rate (0 < alpha <= 1). Higher = faster reaction, less memory. Default: 0.3"
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="Use GPU acceleration (default: True)"
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU acceleration (force CPU)"
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=2,
        help="Number of GPUs to use (default: 2 for A100 setup)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original file before modifying"
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Path to log file (default: logs/add_decay_feature_TIMESTAMP.log)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for checkpoint files (default: checkpoints/)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint if available"
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=float,
        default=5.0,
        help="Checkpoint frequency in percent (default: 5.0 for every 5%%)"
    )
    
    args = parser.parse_args()
    
    # Import time at the start of main()
    import time
    
    # Setup logging first
    logger, log_file = setup_logging(args.log_file)
    
    # Validate alpha
    if args.alpha <= 0 or args.alpha > 1:
        logger.error(f"Invalid alpha value: {args.alpha}. Must be in (0, 1]")
        raise ValueError(f"alpha must be in (0, 1], got {args.alpha}")
    
    logger.info("=" * 80)
    logger.info("Starting decay feature computation")
    logger.info(f"Panel: {args.panel}")
    logger.info(f"Alpha: {args.alpha}")
    logger.info(f"GPUs: {args.n_gpus}")
    logger.info("=" * 80)
    
    # Determine if we should use GPU
    use_gpu = args.use_gpu and not args.no_gpu and GPU_AVAILABLE
    if not GPU_AVAILABLE:
        logger.error("GPU libraries not available")
        raise RuntimeError("GPU libraries (cudf, dask_cudf) are required but not available. Please install them with: pip install cudf-cu12 dask-cudf-cu12 dask-cuda --extra-index-url=https://pypi.nvidia.com")
    if not use_gpu:
        logger.error("GPU acceleration required but disabled")
        raise RuntimeError("GPU acceleration is required. Use --use_gpu (default) or remove --no_gpu flag.")
    logger.info(f"GPU acceleration enabled with {args.n_gpus} GPUs (REQUIRED)")
    
    panel_path = args.panel
    
    if not panel_path.exists():
        logger.error(f"Panel file not found: {panel_path}")
        raise FileNotFoundError(f"Panel file not found: {panel_path}")
    
    # Create backup if requested
    if args.backup:
        backup_path = panel_path.with_suffix(".parquet.backup")
        logger.info(f"Creating backup: {backup_path}")
        import shutil
        shutil.copy2(panel_path, backup_path)
        logger.info(f"Backup created: {backup_path}")
    
    # Setup Dask cluster - REQUIRED for GPU
    if not use_gpu:
        raise RuntimeError("GPU mode is required. Cannot proceed without GPU.")
    
    logger.info("Setting up Dask cluster (REQUIRED for GPU processing)...")
    max_retries = 3
    client = None
    cluster = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Cluster setup attempt {attempt + 1}/{max_retries}")
            client, cluster = setup_dask_cluster(n_gpus=args.n_gpus)
            logger.info("Dask cluster setup successful")
            break
        except Exception as e:
            logger.warning(f"Cluster setup attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying cluster setup in 2 seconds...")
                time.sleep(2)
            else:
                logger.error(f"Failed to setup Dask cluster after {max_retries} attempts")
                raise RuntimeError(f"Failed to setup Dask cluster after {max_retries} attempts: {e}")
    
    try:
        # Load panel
        logger.info(f"Reading panel from {panel_path}")
        load_start = time.time()
        if use_gpu:
            # Load directly to GPU using dask_cudf with optimized settings
            # Use smaller chunks to avoid memory issues
            logger.info("Loading to GPU with dask_cudf...")
            logger.info(f"  File size: {panel_path.stat().st_size / 1024**3:.2f} GB")
            logger.info(f"  Target chunk size: 256MB per partition")
            panel = dask_cudf.read_parquet(
                panel_path,
                chunksize="256MB",  # Read in smaller chunks
                aggregate_files=False,  # Don't aggregate files to reduce memory
            )
            load_elapsed = time.time() - load_start
            logger.info(f"✓ Loaded to GPU (dask_cudf) with {panel.npartitions} partitions in {load_elapsed:.2f}s")
            logger.info(f"  Estimated total rows: {panel.npartitions * 2_000_000:,} (estimate)")
            logger.info(f"  Columns: {list(panel.columns)}")
        else:
            panel = pd.read_parquet(panel_path)
            logger.info(f"Loaded {len(panel):,} rows, {panel['gram'].nunique():,} unique words")
            logger.info(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
            logger.info(f"Columns: {list(panel.columns)}")
        
        # Add decay feature
        if use_gpu:
            logger.info("Starting decay computation on GPU...")
            import time
            start_time = time.time()
            
            panel = add_decay_feature_gpu(panel, alpha=args.alpha, n_gpus=args.n_gpus)
            
            # Compute result with progress tracking and checkpointing
            logger.info("Computing decay on GPU with progress tracking...")
            
            # Get total number of partitions
            total_partitions = panel.npartitions
            logger.info(f"Total partitions to process: {total_partitions}")
            
            # Create checkpoint directory
            checkpoint_dir = args.checkpoint_dir
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Check for existing checkpoint if resuming
            all_results = []
            completed_count = 0
            checkpoint_freq_pct = args.checkpoint_freq
            checkpoint_every = max(1, int(total_partitions * (checkpoint_freq_pct / 100.0)))
            
            # Always check for checkpoints, resume if --resume flag or if checkpoint_latest exists
            latest_checkpoint = checkpoint_dir / "checkpoint_latest.parquet"
            checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*pct.parquet"))
            checkpoint_dataset_dir = checkpoint_dir / "checkpoint_dataset"
            
            # Check for existing partition files (most reliable way to determine progress)
            completed_count = 0
            if checkpoint_dataset_dir.exists():
                import glob
                existing_partitions = sorted(glob.glob(str(checkpoint_dataset_dir / "partition_*.parquet")))
                if existing_partitions:
                    # Extract partition numbers from filenames
                    partition_numbers = []
                    for f in existing_partitions:
                        try:
                            # Extract number from "partition_XXXX.parquet"
                            num_str = Path(f).stem.split("_")[1]
                            partition_numbers.append(int(num_str))
                        except:
                            pass
                    
                    if partition_numbers:
                        completed_count = max(partition_numbers)
                        checkpoint_pct = (completed_count / total_partitions) * 100
                        logger.info(f"Found {len(existing_partitions)} existing partition files")
                        logger.info(f"Resuming from partition {completed_count}/{total_partitions} (~{checkpoint_pct:.1f}%)")
            
            if args.resume or latest_checkpoint.exists() or completed_count > 0:
                logger.info("Checking for existing checkpoints...")
                
                # If we have partition files, use those (most accurate)
                if completed_count > 0:
                    logger.info(f"✓ Resuming from partition {completed_count}/{total_partitions} based on existing partition files")
                    logger.info(f"  Existing partitions will be used in final combination")
                # Otherwise, check for consolidated checkpoint files
                elif latest_checkpoint.exists():
                    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
                    checkpoint_to_load = latest_checkpoint
                    # Try to find corresponding percentage checkpoint for progress calculation
                    if checkpoint_files:
                        latest_pct_checkpoint = checkpoint_files[-1]
                        pct_str = latest_pct_checkpoint.stem.split("_")[1].replace("pct", "")
                        checkpoint_pct = float(pct_str)
                        completed_count = int((checkpoint_pct / 100.0) * total_partitions)
                        logger.info(f"Resuming from ~{checkpoint_pct:.1f}% ({completed_count}/{total_partitions} partitions) based on checkpoint file")
                    else:
                        logger.warning("Could not determine checkpoint percentage, will estimate from file")
                        completed_count = 0
                elif checkpoint_files:
                    checkpoint_to_load = checkpoint_files[-1]
                    pct_str = checkpoint_to_load.stem.split("_")[1].replace("pct", "")
                    checkpoint_pct = float(pct_str)
                    completed_count = int((checkpoint_pct / 100.0) * total_partitions)
                    logger.info(f"Found checkpoint at {checkpoint_pct}%: {checkpoint_to_load}")
                    logger.info(f"Resuming from ~{checkpoint_pct:.1f}% ({completed_count}/{total_partitions} partitions)")
                else:
                    logger.info("No checkpoints found, starting from beginning")
                    completed_count = 0
            elif checkpoint_files:
                logger.info(f"Found {len(checkpoint_files)} checkpoint(s) but --resume not specified. Use --resume to continue from checkpoint.")
            
            # Use Dask's built-in progress tracking and compute with a callback for checkpointing
            if client is not None:
                import cudf
                from dask.distributed import progress, as_completed
                
                logger.info("Computing all partitions with progress tracking...")
                logger.info(f"Checkpoints will be saved every {checkpoint_freq_pct}% ({checkpoint_every} partitions)")
                
                # Convert to delayed objects for each partition to track progress
                logger.info("Converting to delayed objects for progress tracking...")
                delayed_parts = panel.to_delayed()
                logger.info(f"Created {len(delayed_parts)} delayed partition objects")
                
                # Skip already completed partitions if resuming
                if completed_count > 0:
                    logger.info(f"Skipping first {completed_count} partitions (already completed)")
                    delayed_parts = delayed_parts[completed_count:]
                    futures_to_compute = [client.compute(d) for d in delayed_parts]
                else:
                    # Compute all delayed objects as futures
                    logger.info("Submitting all partitions for computation...")
                    futures_to_compute = [client.compute(d) for d in delayed_parts]
                
                logger.info(f"Submitted {len(futures_to_compute)} futures to cluster")
                logger.info(f"Tracking progress - checkpoints every {checkpoint_every} partitions ({checkpoint_freq_pct}%)")
                logger.info("=" * 80)
                
                start_compute = time.time()
                last_progress_time = start_compute
                heartbeat_interval = 30  # Log heartbeat every 30 seconds if no progress
                
                # Process futures as they complete
                import gc
                
                logger.info("Starting to process futures - waiting for first result...")
                logger.info("(This may take a few minutes for initial partitions to complete)")
                logger.info(f"  Total futures submitted: {len(futures_to_compute)}")
                logger.info(f"  Dask dashboard: {client.dashboard_link if hasattr(client, 'dashboard_link') else 'N/A'}")
                
                # Log initial cluster status
                try:
                    workers = client.scheduler_info()['workers']
                    logger.info(f"  Active workers: {len(workers)}")
                    for worker_id, worker_info in list(workers.items())[:5]:  # Show first 5 workers
                        logger.info(f"    Worker {worker_id}: {worker_info.get('nthreads', 'N/A')} threads")
                except:
                    pass
                
                try:
                    future_iterator = as_completed(futures_to_compute, with_results=True)
                    for future in future_iterator:
                        current_time = time.time()
                        time_since_last = current_time - last_progress_time
                        
                        completed_count += 1
                        progress_pct = (completed_count / total_partitions) * 100
                        
                        try:
                            result = future[1]  # Get the result from (future, result) tuple
                            all_results.append(result)
                            
                            result_rows = len(result)
                            result_words = result.index.nunique() if hasattr(result.index, 'nunique') else result['gram'].nunique() if 'gram' in result.columns else 0
                            
                            logger.info(f"✓ Partition {completed_count}/{total_partitions} ({progress_pct:.1f}%) - {result_rows:,} rows, ~{result_words:,} words (waited {time_since_last:.1f}s)")
                            last_progress_time = current_time
                            
                            # Log if we're making progress
                            if completed_count == 1:
                                elapsed_since_start = current_time - start_compute
                                logger.info(f"✓ First partition completed in {elapsed_since_start:.1f}s! Processing is working.")
                                logger.info(f"  Estimated time per partition: {elapsed_since_start:.1f}s")
                                estimated_total = elapsed_since_start * total_partitions
                                logger.info(f"  Rough estimate: {estimated_total/60:.1f} minutes total")
                            elif time_since_last > heartbeat_interval:
                                logger.warning(f"⚠ Long gap ({time_since_last:.1f}s) since last completed partition")
                            
                            # Log every partition for first 10, then every 5
                            if completed_count <= 10 or completed_count % 5 == 0:
                                logger.info(f"  Progress: {progress_pct:.2f}% | {completed_count}/{total_partitions} partitions")
                            
                            # Force periodic garbage collection to prevent memory buildup
                            if completed_count % 10 == 0:
                                gc.collect()
                                logger.debug("Garbage collection performed")
                            
                            # Checkpoint more frequently (every checkpoint_freq_pct%)
                            # Check if we should checkpoint (every N partitions or at milestones)
                            should_checkpoint = (
                                completed_count % checkpoint_every == 0 or 
                                completed_count == total_partitions or
                                int(progress_pct) % int(checkpoint_freq_pct) == 0
                            )
                            
                            # CRITICAL FIX: Write per-partition outputs directly to disk
                            # Save each partition as it completes to avoid RAM accumulation
                            if completed_count == 1:
                                # Initialize checkpoint dataset directory on first partition
                                checkpoint_dataset_dir = checkpoint_dir / "checkpoint_dataset"
                                checkpoint_dataset_dir.mkdir(exist_ok=True)
                                logger.info(f"Initialized checkpoint dataset directory: {checkpoint_dataset_dir}")
                            
                            # Save this partition immediately to disk (no RAM accumulation)
                            partition_file = checkpoint_dataset_dir / f"partition_{completed_count:04d}.parquet"
                            save_part_start = time.time()
                            try:
                                result.to_parquet(partition_file, index=False)
                                save_part_elapsed = time.time() - save_part_start
                                file_size_mb = partition_file.stat().st_size / 1024**2
                                logger.info(f"  💾 Saved partition {completed_count} to disk: {file_size_mb:.1f} MB in {save_part_elapsed:.2f}s")
                            except Exception as e:
                                logger.warning(f"  ✗ Failed to save partition {completed_count}: {e}")
                            
                            if should_checkpoint:
                                logger.info(f"CHECKPOINT: Creating consolidated checkpoint at {progress_pct:.1f}% ({completed_count}/{total_partitions})...")
                                checkpoint_file = checkpoint_dir / f"checkpoint_{int(progress_pct)}pct.parquet"
                                checkpoint_start = time.time()
                                
                                try:
                                    # Read all saved partitions from disk and combine
                                    import glob
                                    partition_files = sorted(glob.glob(str(checkpoint_dataset_dir / "partition_*.parquet")))
                                    logger.info(f"Reading {len(partition_files)} saved partitions for checkpoint...")
                                    
                                    # CRITICAL FIX: Convert to pandas before concatenation to avoid GPU OOM
                                    # Read and combine partitions in chunks, converting to pandas to free GPU memory
                                    logger.info(f"Reading {len(partition_files)} partitions and converting to pandas for checkpoint...")
                                    import pandas as pd
                                    
                                    if len(partition_files) > 20:
                                        logger.info(f"Large dataset ({len(partition_files)} files), combining in chunks with pandas...")
                                        chunk_size = 10
                                        combined_chunks_pd = []
                                        for i in range(0, len(partition_files), chunk_size):
                                            chunk_files = partition_files[i:i+chunk_size]
                                            logger.info(f"  Reading chunk {i//chunk_size + 1}/{(len(partition_files) + chunk_size - 1)//chunk_size}...")
                                            # Read as cuDF, convert to pandas immediately to free GPU memory
                                            chunk_dfs_pd = [cudf.read_parquet(f).to_pandas() for f in chunk_files]
                                            if chunk_dfs_pd:
                                                combined_pd = pd.concat(chunk_dfs_pd, ignore_index=True)
                                                combined_chunks_pd.append(combined_pd)
                                                del chunk_dfs_pd
                                                logger.debug(f"  Chunk {i//chunk_size + 1} converted to pandas: {len(combined_pd):,} rows")
                                        
                                        if combined_chunks_pd:
                                            checkpoint_df_pd = pd.concat(combined_chunks_pd, ignore_index=True)
                                            del combined_chunks_pd
                                            logger.info(f"Combined pandas DataFrame: {len(checkpoint_df_pd):,} rows, {checkpoint_df_pd['gram'].nunique():,} words")
                                            
                                            # Save directly as pandas parquet to avoid GPU memory issues
                                            logger.info(f"Saving consolidated checkpoint as pandas parquet ({len(checkpoint_df_pd):,} rows, {checkpoint_df_pd['gram'].nunique():,} words)...")
                                            checkpoint_df_pd.to_parquet(checkpoint_file, index=False)
                                            checkpoint_elapsed = time.time() - checkpoint_start
                                            logger.info(f"✓ Checkpoint saved to {checkpoint_file} in {checkpoint_elapsed:.2f}s")
                                            
                                            # Skip cuDF conversion - checkpoint is already saved
                                            checkpoint_df = None
                                        else:
                                            checkpoint_df = cudf.concat([cudf.read_parquet(f) for f in partition_files], ignore_index=True)
                                            logger.info(f"Saving consolidated checkpoint ({len(checkpoint_df):,} rows, {checkpoint_df['gram'].nunique():,} words)...")
                                            checkpoint_df.to_parquet(checkpoint_file, index=False)
                                            checkpoint_elapsed = time.time() - checkpoint_start
                                            logger.info(f"✓ Checkpoint saved to {checkpoint_file} in {checkpoint_elapsed:.2f}s ({checkpoint_df.memory_usage(deep=True).sum() / 1024**3:.2f} GB)")
                                    else:
                                        # For small datasets, read directly
                                        checkpoint_df = cudf.concat([cudf.read_parquet(f) for f in partition_files], ignore_index=True)
                                        logger.info(f"Saving consolidated checkpoint ({len(checkpoint_df):,} rows, {checkpoint_df['gram'].nunique():,} words)...")
                                        checkpoint_df.to_parquet(checkpoint_file, index=False)
                                        checkpoint_elapsed = time.time() - checkpoint_start
                                        logger.info(f"✓ Checkpoint saved to {checkpoint_file} in {checkpoint_elapsed:.2f}s ({checkpoint_df.memory_usage(deep=True).sum() / 1024**3:.2f} GB)")
                                    
                                    # Also save a "latest" checkpoint (copy the file instead of recomputing)
                                    import shutil
                                    latest_checkpoint = checkpoint_dir / "checkpoint_latest.parquet"
                                    shutil.copy2(checkpoint_file, latest_checkpoint)
                                    logger.info(f"✓ Latest checkpoint updated: {latest_checkpoint}")
                                    
                                    # Force garbage collection after checkpoint
                                    del checkpoint_df
                                    gc.collect()
                                    logger.info("Memory freed after checkpoint")
                                    
                                except Exception as e:
                                    logger.error(f"✗ Checkpoint consolidation failed: {e}", exc_info=True)
                                    # Don't raise - continue processing even if checkpoint fails
                                    # Individual partitions are already saved
                        
                            # Log progress and memory every 5 partitions
                            if completed_count % 5 == 0 or completed_count == total_partitions:
                                elapsed = time.time() - start_compute
                                avg_time = elapsed / completed_count if completed_count > 0 else 0
                                remaining = avg_time * (total_partitions - completed_count) if avg_time > 0 else 0
                                rate = completed_count / elapsed if elapsed > 0 else 0
                                
                                logger.info("=" * 60)
                                logger.info(f"📊 PROGRESS UPDATE")
                                logger.info(f"  Completed: {completed_count}/{total_partitions} partitions ({progress_pct:.1f}%)")
                                logger.info(f"  Elapsed: {elapsed/60:.1f} minutes ({elapsed:.1f}s)")
                                logger.info(f"  Rate: {rate:.2f} partitions/second")
                                logger.info(f"  Avg time per partition: {avg_time:.1f}s")
                                logger.info(f"  Estimated remaining: {remaining/60:.1f} minutes ({remaining:.1f}s)")
                                logger.info(f"  Estimated completion: {time.strftime('%H:%M:%S', time.localtime(time.time() + remaining))}")
                                logger.info("=" * 60)
                                
                                # Log memory
                                try:
                                    import psutil
                                    import subprocess
                                    process = psutil.Process()
                                    mem_info = process.memory_info()
                                    mem_vms = process.memory_info().vms / 1024**3
                                    mem_percent = process.memory_percent()
                                    logger.info(f"💻 CPU Memory: RSS={mem_info.rss / 1024**3:.2f} GB, VMS={mem_vms:.2f} GB ({mem_percent:.1f}%)")
                                    
                                    # Get system memory info
                                    system_mem = psutil.virtual_memory()
                                    logger.info(f"  System total: {system_mem.total / 1024**3:.1f} GB, available: {system_mem.available / 1024**3:.1f} GB, used: {system_mem.percent:.1f}%")
                                    
                                    try:
                                        result = subprocess.run(
                                            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", 
                                             "--format=csv,noheader,nounits"],
                                            capture_output=True, text=True, timeout=2
                                        )
                                        if result.returncode == 0:
                                            gpu_lines = result.stdout.strip().split('\n')
                                            logger.info(f"🎮 GPU Status:")
                                            for line in gpu_lines:
                                                parts = line.split(', ')
                                                if len(parts) >= 4:
                                                    gpu_id = parts[0]
                                                    mem_used = int(parts[1])
                                                    mem_total = int(parts[2])
                                                    gpu_util = parts[3]
                                                    mem_used_gb = mem_used / 1024
                                                    mem_total_gb = mem_total / 1024
                                                    mem_pct = (mem_used / mem_total) * 100
                                                    logger.info(f"  GPU {gpu_id}: {mem_used_gb:.1f}/{mem_total_gb:.1f} GB ({mem_pct:.1f}%) | Util: {gpu_util}%")
                                    except Exception as e:
                                        logger.debug(f"Could not get GPU info: {e}")
                                except Exception as e:
                                    logger.debug(f"Could not get memory info: {e}")
                                    
                        except Exception as e:
                            logger.error(f"✗ Partition {completed_count} failed: {e}", exc_info=True)
                            raise
                        
                        # Save emergency checkpoint if we have results but no regular checkpoint saved recently
                        # This prevents total loss if process crashes
                        if len(all_results) > 0 and completed_count % (checkpoint_every * 2) == checkpoint_every:
                            try:
                                logger.info("Emergency checkpoint: Saving progress...")
                                emergency_file = checkpoint_dir / "checkpoint_emergency.parquet"
                                if len(all_results) <= 20:
                                    emergency_df = cudf.concat(all_results, ignore_index=True)
                                else:
                                    # Combine in chunks
                                    chunks = []
                                    for i in range(0, len(all_results), 10):
                                        chunk = all_results[i:i+10]
                                        if chunk:
                                            chunks.append(cudf.concat(chunk, ignore_index=True))
                                    emergency_df = cudf.concat(chunks, ignore_index=True) if chunks else cudf.concat(all_results, ignore_index=True)
                                
                                emergency_df.to_parquet(emergency_file, index=False)
                                logger.info(f"Emergency checkpoint saved: {emergency_file} ({len(emergency_df):,} rows)")
                                del emergency_df
                                gc.collect()
                            except Exception as e:
                                logger.warning(f"Emergency checkpoint failed: {e}")
                                
                except KeyboardInterrupt:
                    logger.warning("Interrupted by user - saving emergency checkpoint...")
                    # Save whatever we have
                    if all_results:
                        try:
                            emergency_file = checkpoint_dir / "checkpoint_interrupted.parquet"
                            emergency_df = cudf.concat(all_results, ignore_index=True)
                            emergency_df.to_parquet(emergency_file, index=False)
                            logger.info(f"Interrupted checkpoint saved: {emergency_file} ({len(emergency_df):,} rows)")
                            raise
                        except Exception as e:
                            logger.error(f"Failed to save interrupted checkpoint: {e}")
                            raise
                except Exception as e:
                    logger.error(f"Error during processing: {e}", exc_info=True)
                    # Try to save emergency checkpoint
                    if all_results:
                        try:
                            emergency_file = checkpoint_dir / "checkpoint_error.parquet"
                            emergency_df = cudf.concat(all_results, ignore_index=True)
                            emergency_df.to_parquet(emergency_file, index=False)
                            logger.info(f"Error checkpoint saved: {emergency_file} ({len(emergency_df):,} rows)")
                        except:
                            pass
                    raise
                
                # CRITICAL FIX: Read all saved partitions from disk instead of concatenating in RAM
                logger.info("=" * 80)
                logger.info("FINAL COMBINATION: Reading all saved partitions from disk...")
                final_start = time.time()
                
                import glob
                checkpoint_dataset_dir = checkpoint_dir / "checkpoint_dataset"
                partition_files = sorted(glob.glob(str(checkpoint_dataset_dir / "partition_*.parquet")))
                
                logger.info(f"Found {len(partition_files)} partition files to combine")
                if partition_files:
                    total_size_mb = sum(Path(f).stat().st_size for f in partition_files) / 1024**2
                    logger.info(f"Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
                
                if partition_files:
                    logger.info(f"Reading {len(partition_files)} partitions from disk...")
                    # CRITICAL FIX: Convert to pandas before concatenation to avoid GPU OOM
                    # Read and combine in chunks, converting to pandas to free GPU memory
                    import pandas as pd
                    
                    if len(partition_files) > 20:
                        logger.info(f"Large dataset ({len(partition_files)} files), combining in chunks with pandas conversion...")
                        chunk_size = 10
                        combined_chunks_pd = []
                        for i in range(0, len(partition_files), chunk_size):
                            chunk_files = partition_files[i:i+chunk_size]
                            logger.info(f"  Reading chunk {i//chunk_size + 1}/{(len(partition_files) + chunk_size - 1)//chunk_size}...")
                            # Read as cuDF, convert to pandas immediately to free GPU memory
                            chunk_dfs_pd = [cudf.read_parquet(f).to_pandas() for f in chunk_files]
                            if chunk_dfs_pd:
                                combined_pd = pd.concat(chunk_dfs_pd, ignore_index=True)
                                combined_chunks_pd.append(combined_pd)
                                del chunk_dfs_pd
                                logger.info(f"  Chunk {i//chunk_size + 1} converted to pandas: {len(combined_pd):,} rows")
                        
                        if combined_chunks_pd:
                            logger.info(f"Combining {len(combined_chunks_pd)} pandas chunks...")
                            panel_pd = pd.concat(combined_chunks_pd, ignore_index=True)
                            del combined_chunks_pd
                            logger.info(f"Final pandas DataFrame: {len(panel_pd):,} rows, {panel_pd['gram'].nunique():,} words")
                            # Keep as pandas for now - will convert to cuDF only when needed for final save
                            # This avoids GPU memory pressure during final combination
                            panel = panel_pd  # Store as pandas for now
                        else:
                            panel = cudf.concat([cudf.read_parquet(f) for f in partition_files], ignore_index=True)
                    else:
                        # For small datasets, read directly
                        panel = cudf.concat([cudf.read_parquet(f) for f in partition_files], ignore_index=True)
                    
                    final_elapsed = time.time() - final_start
                    logger.info(f"✓ Final combination in {final_elapsed:.2f}s")
                    logger.info(f"Final: {len(panel):,} rows, {panel['gram'].nunique():,} words")
                    logger.info(f"Memory: {panel.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
                elif all_results:
                    # Fallback: if no files saved, use in-memory results
                    logger.warning("No partition files found, using in-memory results...")
                    logger.info(f"Combining {len(all_results)} partitions from memory...")
                    panel = cudf.concat(all_results, ignore_index=True)
                    final_elapsed = time.time() - final_start
                    logger.info(f"✓ Final combination in {final_elapsed:.2f}s")
                    logger.info(f"Final: {len(panel):,} rows, {panel['gram'].nunique():,} words")
                    logger.info(f"Memory: {panel.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
                    del all_results
                else:
                    logger.error("No results available!")
                    raise RuntimeError("No partition results available")
                
                logger.info("=" * 80)
            else:
                # Fallback: compute all at once
                logger.info("Computing all partitions (no checkpointing)...")
                panel = panel.compute()
            
            elapsed = time.time() - start_time
            logger.info(f"Computed decay in {elapsed:.2f} seconds")
            logger.info(f"Results: {len(panel):,} rows, {panel['gram'].nunique():,} unique words")
        else:
            logger.info("Starting decay computation on CPU...")
            import time
            start_time = time.time()
            panel = add_decay_feature_cpu(panel, alpha=args.alpha)
            elapsed = time.time() - start_time
            logger.info(f"Computed decay in {elapsed:.2f} seconds")
        
        # Verify decay column was added
        if "tf_share_decay" not in panel.columns:
            logger.error("Failed to add tf_share_decay column")
            raise RuntimeError("Failed to add tf_share_decay column")
        
        logger.info("Decay column successfully added")
        
        # Show some statistics
        logger.info("Computing statistics for tf_share_decay...")
        import pandas as pd
        if isinstance(panel, pd.DataFrame):
            # pandas DataFrame (from final combination)
            mean_val = float(panel['tf_share_decay'].mean())
            std_val = float(panel['tf_share_decay'].std())
            min_val = float(panel['tf_share_decay'].min())
            max_val = float(panel['tf_share_decay'].max())
            non_zero = int((panel['tf_share_decay'] > 0).sum())
            total = len(panel)
        elif use_gpu:
            # CRITICAL FIX: Use cuDF/Dask-cuDF operations directly, avoid to_pandas()
            # If panel is dask_cudf, compute aggregates; if cuDF, use directly
            if hasattr(panel, 'compute'):
                # dask_cudf DataFrame
                mean_val = float(panel['tf_share_decay'].mean().compute())
                std_val = float(panel['tf_share_decay'].std().compute())
                min_val = float(panel['tf_share_decay'].min().compute())
                max_val = float(panel['tf_share_decay'].max().compute())
                non_zero = int((panel['tf_share_decay'] > 0).sum().compute())
                total = int(len(panel))
            else:
                # cuDF DataFrame
                mean_val = float(panel['tf_share_decay'].mean())
                std_val = float(panel['tf_share_decay'].std())
                min_val = float(panel['tf_share_decay'].min())
                max_val = float(panel['tf_share_decay'].max())
                non_zero = int((panel['tf_share_decay'] > 0).sum())
                total = len(panel)
        else:
            mean_val = panel['tf_share_decay'].mean()
            std_val = panel['tf_share_decay'].std()
            min_val = panel['tf_share_decay'].min()
            max_val = panel['tf_share_decay'].max()
            non_zero = (panel['tf_share_decay'] > 0).sum()
            total = len(panel)
        
        logger.info(f"tf_share_decay statistics:")
        logger.info(f"  - Mean: {mean_val:.6e}")
        logger.info(f"  - Std: {std_val:.6e}")
        logger.info(f"  - Min: {min_val:.6e}")
        logger.info(f"  - Max: {max_val:.6e}")
        logger.info(f"  - Non-zero: {non_zero:,} / {total:,}")
        
        # Save updated panel
        logger.info(f"Writing updated panel to {panel_path}")
        save_start = time.time()
        
        # Check if panel is pandas (from final combination with pandas conversion)
        import pandas as pd
        if isinstance(panel, pd.DataFrame):
            logger.info("Panel is pandas DataFrame, saving directly...")
            panel.to_parquet(panel_path, index=False)
        elif use_gpu:
            # CRITICAL FIX: Use cuDF/Dask-cuDF writer directly, avoid to_pandas()
            if hasattr(panel, 'to_parquet'):
                # dask_cudf DataFrame - save as parquet dataset
                output_dir = str(panel_path).replace('.parquet', '_with_decay')
                logger.info(f"Saving dask_cudf DataFrame as parquet dataset to {output_dir}/...")
                panel.to_parquet(output_dir, write_index=False)
                logger.info(f"Saved as dataset. To create single file, run: python -c \"import cudf; cudf.read_parquet('{output_dir}').to_parquet('{panel_path}', index=False)\"")
                # Also save directly as single file if we can compute it
                logger.info("Computing final result and saving as single file...")
                panel_final = panel.compute()
                panel_final.to_parquet(panel_path, index=False)
                logger.info(f"Saved single file: {panel_path}")
            else:
                # cuDF DataFrame
                panel.to_parquet(panel_path, index=False)
        else:
            panel.to_parquet(panel_path, index=False)
        save_elapsed = time.time() - save_start
        logger.info(f"Saved panel in {save_elapsed:.2f} seconds")
        
        logger.info("=" * 80)
        logger.info(f"SUCCESS: Added tf_share_decay column (α={args.alpha})")
        if use_gpu:
            logger.info(f"Panel: {len(panel):,} rows, {len(panel.columns)} columns (GPU)")
        else:
            logger.info(f"Panel: {len(panel):,} rows, {len(panel.columns)} columns")
        logger.info(f"Log file: {log_file}")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        raise
    finally:
        # Cleanup Dask cluster
        if client is not None:
            logger.info("Closing Dask cluster...")
            try:
                client.close()
                cluster.close()
                logger.info("Dask cluster closed successfully")
            except Exception as e:
                logger.warning(f"Error closing cluster: {e}")


if __name__ == "__main__":
    main()


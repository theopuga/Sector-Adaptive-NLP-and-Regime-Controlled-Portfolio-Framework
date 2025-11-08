#!/usr/bin/env python3
"""
Multi-instance runner for FinBERT analysis.
Splits shards across multiple independent processes for maximum parallelization.
Useful for systems with many CPU cores (e.g., Ryzen 9 7900X with 24 threads).
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
import random
import pandas as pd
import tempfile
import polars as pl

def discover_shards(shards_dir: str, limit: int = 0) -> List[str]:
    """Discover all parquet shards in directory."""
    from glob import glob
    shards = sorted(glob(str(Path(shards_dir) / "**/*.parquet")))
    if limit > 0:
        shards = shards[:limit]
    return shards

def split_shards(shards: List[str], n_instances: int) -> List[List[str]]:
    """Split shard list into n roughly equal partitions."""
    chunks = [[] for _ in range(n_instances)]
    for i, shard in enumerate(shards):
        chunks[i % n_instances].append(shard)
    return chunks

def main():
    ap = argparse.ArgumentParser(description="Run multiple FinBERT instances in parallel")
    
    # Core arguments
    ap.add_argument("--script", default="run/sec_finbert_monthly_optimized.py",
                    help="Path to the FinBERT script")
    ap.add_argument("--instances", type=int, default=4,
                    help="Number of independent instances to run (default: 4)")
    ap.add_argument("--worker_per_instance", type=int, default=4,
                    help="Workers per instance (default: 4)")
    
    # Input/output
    ap.add_argument("--manifest", default="")
    ap.add_argument("--shards_dir", default="")
    ap.add_argument("--scope", default="global")
    ap.add_argument("--scope_value", default="")
    ap.add_argument("--limit_shards", type=int, default=0)
    
    # FinBERT args
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--lag_days", type=int, default=7)
    ap.add_argument("--fp16", action="store_true", help="Use FP16 precision (faster, less memory)")
    ap.add_argument("--checkpoint_dir", default="data/textsec/processed/FinBert/checkpoints")
    
    # Output
    ap.add_argument("--out_base", default="data/textsec/processed/FinBert/instances", help="Base directory for per-instance outputs")
    ap.add_argument("--final_out", default="data/textsec/processed/FinBert/finbert_master.parquet", help="Final merged output file path")
    
    args = ap.parse_args()
    
    # Discover shards
    if args.manifest:
        man = pd.read_csv(args.manifest)
        col = "path" if "path" in man.columns else ("shard_path" if "shard_path" in man.columns else None)
        if not col:
            raise ValueError("Manifest must have 'path' or 'shard_path'")
        shards = man[col].astype(str).tolist()
    elif args.shards_dir:
        shards = discover_shards(args.shards_dir, args.limit_shards)
    else:
        print("❌ Must specify either --manifest or --shards_dir")
        sys.exit(1)
    
    print(f"📊 Total shards discovered: {len(shards)}")
    
    # Split into instance groups
    shard_groups = split_shards(shards, args.instances)
    
    print(f"🚀 Launching {args.instances} instances...")
    print(f"   Workers per instance: {args.worker_per_instance}")
    print(f"   Total workers: {args.instances * args.worker_per_instance}")
    
    # Launch processes
    processes = []
    out_dirs = []
    
    for i, group in enumerate(shard_groups):
        if not group:
            print(f"  ⚠️  Instance {i+1} has no shards, skipping")
            continue
        
        out_dir = Path(args.out_base) / f"instance_{i+1}"
        out_dirs.append(out_dir)
        
        # Create a temporary manifest for this instance
        tmp_manifest = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        pd.DataFrame({"path": group}).to_csv(tmp_manifest.name, index=False)
        tmp_manifest.close()
        
        cmd = [
            sys.executable, args.script,
            "--manifest", tmp_manifest.name,
            "--num_workers", str(args.worker_per_instance),
            "--device", args.device,
            "--batch_size", str(args.batch_size),
            "--max_length", str(args.max_length),
            "--lag_days", str(args.lag_days),
            "--checkpoint_dir", str(out_dir / "checkpoints"),
            "--out", str(out_dir / "result.parquet"),
        ]
        
        if args.fp16:
            cmd.append("--fp16")
        
        if args.scope != "global":
            cmd.extend(["--scope", args.scope])
            if args.scope_value:
                cmd.extend(["--scope_value", args.scope_value])
        
        print(f"  🚀 Instance {i+1}: {len(group)} shards → {out_dir / 'result.parquet'}")
        
        proc = subprocess.Popen(cmd)
        processes.append((proc, tmp_manifest.name))
    
    # Wait for all to complete
    print("\n⏳ Waiting for all instances to complete...")
    for i, (proc, tmp_manifest) in enumerate(processes):
        proc.wait()
        Path(tmp_manifest).unlink()  # cleanup
        if proc.returncode == 0:
            print(f"  ✅ Instance {i+1} completed successfully")
        else:
            print(f"  ❌ Instance {i+1} failed with return code {proc.returncode}")
    
    # Merge all results
    print("\n🔗 Merging results from all instances...")
    
    all_results = []
    for out_dir in out_dirs:
        result_file = out_dir / "result.parquet"
        if result_file.exists():
            df = pl.read_parquet(result_file)
            all_results.append(df)
            print(f"  ✅ Loaded {len(df):,} rows from {result_file}")

    if not all_results:
        print("❌ No results to merge!")
        sys.exit(1)

    # Final merge
    final = pl.concat(all_results)

    # Detect duplicates on key; include section if present (USA scope)
    keys = ["gvkey", "iid", "month_end", "country", "continent"]
    if "section" in final.columns:
        keys.append("section")
    
    dup_cnt = (
        final.group_by(keys)
             .len()
             .filter(pl.col("len") > 1)
             .height
    )

    if dup_cnt > 0:
        # Aggregate base metrics with doc-count-weighted means; drop engineered columns
        base_cols = {
            "doc_sent_count",
            "prob_neg_mean",
            "prob_neu_mean",
            "prob_pos_mean",
            "sent_pos_minus_neg_mean",
            "company_name",
            "excntry",
        }
        present_base = [c for c in base_cols if c in final.columns]

        # Weighted averages for prob_*, sent_pos_minus_neg_mean
        wsum = lambda col: (pl.col(col) * pl.col("doc_sent_count")).sum()
        safe_div = lambda num, den: (num / pl.when(den == 0).then(None).otherwise(den))

        agg_exprs = [pl.sum("doc_sent_count").alias("doc_sent_count")]
        if "prob_neg_mean" in present_base:
            agg_exprs.append(safe_div(wsum("prob_neg_mean"), pl.sum("doc_sent_count")).alias("prob_neg_mean"))
        if "prob_neu_mean" in present_base:
            agg_exprs.append(safe_div(wsum("prob_neu_mean"), pl.sum("doc_sent_count")).alias("prob_neu_mean"))
        if "prob_pos_mean" in present_base:
            agg_exprs.append(safe_div(wsum("prob_pos_mean"), pl.sum("doc_sent_count")).alias("prob_pos_mean"))
        if "sent_pos_minus_neg_mean" in present_base:
            agg_exprs.append(safe_div(wsum("sent_pos_minus_neg_mean"), pl.sum("doc_sent_count")).alias("sent_pos_minus_neg_mean"))

        # Preserve identifiers as first
        if "company_name" in present_base:
            agg_exprs.append(pl.first("company_name").alias("company_name"))
        if "excntry" in present_base:
            agg_exprs.append(pl.first("excntry").alias("excntry"))
        if "section" in final.columns:
            agg_exprs.append(pl.first("section").alias("section"))

        sort_cols = ["month_end", "country", "gvkey", "iid"]
        if "section" in final.columns:
            sort_cols.append("section")
        
        final = (
            final.group_by(keys)
                 .agg(agg_exprs)
                 .sort(sort_cols)
        )
    else:
        # Keys are unique across instances; just sort for consistency
        sort_cols = ["month_end", "country", "gvkey", "iid"]
        if "section" in final.columns:
            sort_cols.append("section")
        final = final.sort(sort_cols)

    # Recompute engineered SEC features globally to avoid stale/partial values
    try:
        from run.sec_finbert_monthly_optimized import _compute_sec_features  # type: ignore
        final = _compute_sec_features(final)
    except Exception as e:
        print(f"  ⚠️  global feature recompute skipped: {e}")

    Path(args.final_out).parent.mkdir(parents=True, exist_ok=True)
    final.write_parquet(args.final_out)

    print(f"\n✅ Final result saved: {args.final_out}")
    print(f"   Rows: {len(final):,}")
    print(f"   Firms: {final.select(pl.col(['gvkey','iid'])).unique().height:,}")
    print(f"   Months: {final['month_end'].n_unique():,}")

if __name__ == "__main__":
    main()


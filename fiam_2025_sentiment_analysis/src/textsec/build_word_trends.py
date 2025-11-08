#!/usr/bin/env python3
"""
Build Word Trends Panel
========================
Purpose:
    Constructs a **monthly time series of word frequencies (TF/DF shares)**  
    from cleaned SEC filings, using a **dynamically evolving lexicon**  
    that expands yearly without look-ahead leakage.

Inputs:
    --paths : YAML with paths to filings, stopwords, and processed directory.
    Optional args:
        --seed_start / --seed_end : Years defining baseline lexicon window.
        --min_ngr / --max_ngr : n-gram length range.
        --min_df_frac / --max_df_frac : Baseline document-frequency thresholds.
        --yearly_min_df_abs / --yearly_max_df_frac : Yearly lexicon extension thresholds.
        --export_summaries : If set, exports CSV summaries.

Outputs:
    - `word_trends_monthly.parquet` : Monthly panel with (gram, tf, df, shares, momentum)
    - `word_baseline_lexicon_2005_2010.json` : Seed lexicon
    - Optional summaries: `top_momentum_<year>.csv`, `top_dfshare_<year>.csv`

Core Logic:
    1. Builds seed lexicon on baseline years (e.g. 2005–2010).
    2. Expands lexicon yearly using *only past data*.
    3. Streams filings to build month-level TF/DF metrics.
    4. Adds normalized shares and momentum indicators.
    5. Writes to parquet for downstream sector-mapping scripts.

Guarantees:
    - Fully leak-free by design.
    - Scales efficiently through streaming aggregation.

Command:
python build_word_trends.py --paths config/paths.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable, List, Set, Tuple, Dict

import pandas as pd
import re
import yaml

# --------------------------- I/O helpers ---------------------------

def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(paths_yaml: str | Path) -> dict:
    cfg = _load_yaml(paths_yaml)
    textsec = cfg.get("textsec", {})

    # REQUIRED: filings_clean must exist in YAML
    if "filings_clean" not in textsec:
        raise KeyError("paths.yaml is missing required key: textsec.filings_clean")

    processed_dir = Path(textsec.get("processed_dir", "data/textsec/processed"))
    filings_clean = Path(textsec["filings_clean"])  # canonical input

    stop_fp = cfg.get("files", {}).get("stopwords", "resources/stopwords_en.txt")

    return {
        "processed_dir": processed_dir,
        "filings_clean": filings_clean,
        "stopwords": Path(stop_fp),
    }


# ---------------------- Tokenization utilities ---------------------
_WORD_RX = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str, stop: set[str]) -> List[str]:
    """Tokenize text into lowercased word tokens and filter out stopwords."""
    if not isinstance(text, str):
        text = str(text)
    toks = _WORD_RX.findall(text.lower())
    return [t for t in toks if t not in stop]


def ngrams(tokens: Iterable[str], n: int) -> List[str]:
    """Build whitespace-joined n-grams from a token sequence."""
    toks = list(tokens)
    if n <= 0 or len(toks) < n:
        return []
    return [" ".join(toks[i:i + n]) for i in range(0, len(toks) - n + 1)]


def keep_word(tok: str, stop: set[str]) -> bool:
    """Generic filter: keep tokens of length >=2 (df thresholds applied later)."""
    if tok in stop:
        return False
    return len(tok) >= 2


# ------------------ Generic normalization (agnostic) ----------------
_ACR_RX = re.compile(r"\b(?:[A-Za-z]\.){1,}[A-Za-z]\b")  # e.g., A.I., U.S.


def _collapse_dotted_acronyms(s: str) -> str:
    def _join(m):
        return re.sub(r"\.", "", m.group(0))
    return _ACR_RX.sub(_join, s)


def normalize_text_generic(s: str) -> str:
    """Domain-agnostic normalization (no mapping to any semantic/industry terms)."""
    if not isinstance(s, str):
        s = str(s)
    s = _collapse_dotted_acronyms(s)
    s = re.sub(r"[-_]", " ", s)        # split hyphen/underscore
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


# ---------------------- Lexicon construction -----------------------

def build_baseline_lexicon(
    base_docs: pd.DataFrame,
    stop: set[str],
    min_ngr: int,
    max_ngr: int,
    min_df_frac: float,
    max_df_frac: float,
    out_path: Path,
) -> Set[str]:
    """Build the baseline (seed) lexicon on base_docs using df thresholds."""
    docfreq: Counter[str] = Counter()
    total_docs = 0

    for _, r in base_docs.iterrows():
        txt = normalize_text_generic(r["text"])
        ws = [w for w in tokenize(txt, stop) if keep_word(w, stop)]
        if not ws:
            continue
        total_docs += 1
        grams = set()
        for n in range(min_ngr, max_ngr + 1):
            grams |= set(ngrams(ws, n))
        docfreq.update(grams)

    baseline_lex: List[str]
    if total_docs == 0:
        baseline_lex = []
    else:
        min_df = int(min_df_frac * total_docs)
        max_df = int(max_df_frac * total_docs) if max_df_frac <= 1.0 else int(max_df_frac * total_docs)
        baseline_lex = [g for g, c in docfreq.items() if c >= min_df and c <= max_df]

    out_path.write_text(json.dumps(sorted(baseline_lex), indent=2))
    print(
        f"[INFO] Baseline lexicon saved → {out_path} | terms={len(baseline_lex)} "
        f"(docs={total_docs}, min_df_frac={min_df_frac}, max_df_frac={max_df_frac})"
    )
    return set(baseline_lex)


def extend_lexicon_from_history(
    hist_docs: pd.DataFrame,
    stop: set[str],
    min_ngr: int,
    max_ngr: int,
    min_df_abs: int = 50,
    max_df_frac: float = 0.50,
) -> Set[str]:
    """Discover candidate grams from *historical* docs only (no look-ahead)."""
    docfreq: Counter[str] = Counter()
    total_docs = 0

    for _, r in hist_docs.iterrows():
        txt = normalize_text_generic(r["text"])
        ws = [w for w in tokenize(txt, stop) if keep_word(w, stop)]
        if not ws:
            continue
        total_docs += 1
        grams = set()
        for n in range(min_ngr, max_ngr + 1):
            grams |= set(ngrams(ws, n))
        docfreq.update(grams)

    if total_docs == 0:
        return set()

    max_df = int(max_df_frac * total_docs)
    return {g for g, c in docfreq.items() if c >= min_df_abs and c <= max_df}


def build_lexicon_snapshots(
    all_docs: pd.DataFrame,
    seed_lexicon: Set[str],
    stop: set[str],
    min_ngr: int,
    max_ngr: int,
    yearly_min_df_abs: int,
    yearly_max_df_frac: float,
    seed_end_year: int,
) -> Dict[int, Set[str]]:
    """Create {year -> lexicon_as_of_Jan_year} with past-only extensions."""
    lex_by_year: Dict[int, Set[str]] = {}
    lex_so_far: Set[str] = set(seed_lexicon)

    all_docs = all_docs.sort_values("date").copy()
    all_docs["date"] = pd.to_datetime(all_docs["date"])  # ensure ts

    start_year = seed_end_year + 1
    end_year = int(all_docs["date"].dt.year.max())

    for year in range(start_year, end_year + 1):
        cutoff = pd.Timestamp(f"{year}-01-01")
        hist = all_docs[all_docs["date"] < cutoff]
        ext = extend_lexicon_from_history(
            hist_docs=hist,
            stop=stop,
            min_ngr=min_ngr,
            max_ngr=max_ngr,
            min_df_abs=yearly_min_df_abs,
            max_df_frac=yearly_max_df_frac,
        )
        before = len(lex_so_far)
        lex_so_far |= ext
        lex_by_year[year] = set(lex_so_far)
        print(f"[INFO] Lexicon@{year}-01: +{len(lex_so_far) - before} terms (size={len(lex_so_far)})")

    return lex_by_year


# ------------------------ Panel construction -----------------------

def build_monthly_word_panel_streaming(
    monthly_docs: pd.DataFrame,
    stop: set[str],
    min_ngr: int,
    max_ngr: int,
    seed_lexicon: Set[str],
    lex_by_year: Dict[int, Set[str]],
    flush_every: str = "Q"
):
    """
    Generator that yields small DataFrames of (gram,date,tf,df_docs,month_tokens,month_docs)
    in chronological order, flushing by period (default: quarterly) to keep memory low.
    """
    monthly_docs = monthly_docs.copy()
    monthly_docs["date"] = pd.to_datetime(monthly_docs["date"])  # ensure ts
    monthly_docs.sort_values("date", inplace=True)

    lex0 = set(seed_lexicon)

    # We'll aggregate inside a period bucket (e.g., quarter), then yield & reset
    def _flush_bucket(bucket_maps, bucket_period):
        tf_map, df_map, month_token_total, month_doc_total = bucket_maps
        rows: List[dict] = []
        all_keys = set(tf_map.keys()) | set(df_map.keys())
        for (d, g) in all_keys:
            rows.append({
                "date": d,
                "gram": g,
                "tf": tf_map.get((d, g), 0),
                "df_docs": df_map.get((d, g), 0),
                "month_tokens": month_token_total.get(d, 0),
                "month_docs": month_doc_total.get(d, 0),
            })
        if rows:
            chunk = pd.DataFrame(rows).sort_values(["gram","date"]).reset_index(drop=True)
            print(f"[PANEL] flush {bucket_period}: rows={len(chunk):,}")
            return chunk
        return None

    # Initialize
    tf_map: dict[tuple[pd.Timestamp, str], int] = defaultdict(int)
    df_map: dict[tuple[pd.Timestamp, str], int] = defaultdict(int)
    month_token_total: dict[pd.Timestamp, int] = defaultdict(int)
    month_doc_total: dict[pd.Timestamp, int] = defaultdict(int)

    current_bucket = None

    for _, row in monthly_docs.iterrows():
        date = row["date"]
        # compute bucket key
        if flush_every.upper() == "Q":
            bucket_key = f"{date.year}-Q{((date.month - 1)//3)+1}"
        elif flush_every.upper() == "M":
            bucket_key = f"{date.year}-{date.month:02d}"
        else:  # yearly
            bucket_key = str(date.year)

        if current_bucket is None:
            current_bucket = bucket_key

        # If bucket changed, flush
        if bucket_key != current_bucket:
            chunk = _flush_bucket((tf_map, df_map, month_token_total, month_doc_total), current_bucket)
            # reset maps
            tf_map.clear(); df_map.clear(); month_token_total.clear(); month_doc_total.clear()
            if chunk is not None:
                yield chunk
            current_bucket = bucket_key

        year = int(date.year)
        lex_current = lex_by_year.get(year, lex0)

        txt = normalize_text_generic(row["text"])
        ws = [w for w in tokenize(txt, stop) if keep_word(w, stop)]
        if not ws:
            continue

        month_token_total[date] += len(ws)
        month_doc_total[date] += 1

        grams_all: List[str] = []
        for n in range(min_ngr, max_ngr + 1):
            grams_all += ngrams(ws, n)

        # Filter grams by lexicon snapshot early to reduce memory
        grams_all = [g for g in grams_all if g in lex_current]
        if not grams_all:
            continue
        grams_df = set(grams_all)

        for g in grams_all:
            tf_map[(date, g)] += 1
        for g in grams_df:
            df_map[(date, g)] += 1

    # final flush
    chunk = _flush_bucket((tf_map, df_map, month_token_total, month_doc_total), current_bucket)
    if chunk is not None:
        yield chunk


def add_shares_and_momentum(panel: pd.DataFrame) -> pd.DataFrame:
    """Add tf_share, df_share, and momentum columns to (gram, date) panel.
    Uses groupby.transform (not apply) to avoid MultiIndex alignment issues.
    """
    df = panel.copy()

    # Ensure proper types/order
    df["date"] = pd.to_datetime(df["date"])  # in case concat changed dtype
    df = df.sort_values(["gram", "date"], kind="mergesort").reset_index(drop=True)

    # Shares
    denom_tokens = df["month_tokens"].replace(0, pd.NA)
    denom_docs   = df["month_docs"].replace(0, pd.NA)
    df["tf_share"] = (df["tf"] / denom_tokens).astype("float64")
    df["df_share"] = (df["df_docs"] / denom_docs).astype("float64")

    g = df.groupby("gram", sort=False)

    # Momentum via transform to keep index aligned with df
    df["mom_1m"]  = g["tf_share"].transform(lambda s: s.pct_change(1))
    df["mom_12m"] = g["tf_share"].transform(lambda s: s.pct_change(12))

    def _roll_ret(s: pd.Series, win: int) -> pd.Series:
        return s.divide(s.shift(win)) - 1.0

    df["r3m"]  = g["tf_share"].transform(lambda s: _roll_ret(s, 3))
    df["r6m"]  = g["tf_share"].transform(lambda s: _roll_ret(s, 6))
    df["r12m"] = g["tf_share"].transform(lambda s: _roll_ret(s, 12))

    return df


# --------------------------- Summaries -----------------------------

def export_summaries(panel: pd.DataFrame, out_dir: Path, top_k: int = 50) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top grams by 12m momentum in the last year available
    last_year = int(panel["date"].max().year)
    recent = panel[panel["date"].dt.year == last_year]
    top_mom = (
        recent.sort_values(["mom_12m"], ascending=False)
        .dropna(subset=["mom_12m"])  # avoid NaN early periods
        .head(top_k)
    )
    top_mom.to_csv(out_dir / f"top_momentum_{last_year}.csv", index=False)

    # Most common grams by df_share in the last year
    top_df = (
        recent.sort_values(["df_share"], ascending=False)
        .dropna(subset=["df_share"])
        .head(top_k)
    )
    top_df.to_csv(out_dir / f"top_dfshare_{last_year}.csv", index=False)


# ----------------------------- Main -------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", required=True, help="YAML with input/output paths (must include textsec.filings_clean)")

    # Seed window
    p.add_argument("--seed_start", default="2005-01-01")
    p.add_argument("--seed_end", default="2010-12-31")

    # N-gram settings
    p.add_argument("--min_ngr", type=int, default=1)
    p.add_argument("--max_ngr", type=int, default=2)

    # Baseline lexicon thresholds
    p.add_argument("--min_df_frac", type=float, default=0.001)
    p.add_argument("--max_df_frac", type=float, default=0.50)

    # Yearly extension thresholds
    p.add_argument("--yearly_min_df_abs", type=int, default=50)
    p.add_argument("--yearly_max_df_frac", type=float, default=0.50)

    # Misc
    p.add_argument("--export_summaries", action="store_true")
    return p.parse_args()


def load_stopwords(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {s.strip().lower() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()}


def load_monthly_docs(path: Path) -> pd.DataFrame:
    """Load filings with a usable text column and derive a monthly 'date'.
    Accepts Parquet/CSV. Tries several common date field names and also (year, month) pairs.
    Logs what it used so debugging is easy.
    """
    # 1) Read
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # 2) Identify text column
    text_col = None
    for cand in ("text", "clean_text", "body", "content", "document_text"):
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        raise ValueError(
            "Input docs must have a text column (one of: text, clean_text, body, content, document_text)."
        )

    # 3) Identify/construct a date column (availability). Try several common fields.
    date_col = None
    date_candidates = (
        "date",
        "available_month",
        "available_date",
        "filing_date",
        "filed",
        "file_date",
        "accepted",
        "accepted_date",
        "report_period",
        "period",
        "adsh_date",
    )
    for cand in date_candidates:
        if cand in df.columns:
            date_col = cand
            break

    # (year, month) fallback
    if date_col is None and {"year", "month"}.issubset(df.columns):
        df["date"] = pd.to_datetime(
            df[["year", "month"]].assign(day=1)
            .rename(columns={"year": "Y", "month": "M"})
            .apply(lambda r: f"{int(r['Y']):04d}-{int(r['M']):02d}-01", axis=1)
        )
        date_col = "date"

    if date_col is None:
        raise ValueError(
            "Could not find a date column. Provide one of: date, available_month, available_date, "
            "filing_date, filed, file_date, accepted, accepted_date, report_period, period, adsh_date, "
            "or (year, month). Columns present were: " + ", ".join(map(str, df.columns))
        )

    # 4) Normalize to month start
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise ValueError(f"Date column '{date_col}' could not be parsed to datetime.")

    df = df.rename(columns={date_col: "date", text_col: "text"})
    df = df[["date", "text"]].copy()
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # 5) Basic logging
    print(f"[LOAD] {path} → rows={len(df):,}, cols={list(df.columns)}")
    print(f"[LOAD] using text column '{text_col}', date column '{date_col}' → span {df['date'].min()} → {df['date'].max()}")

    return df


def main():
    args = parse_args()
    paths = resolve_paths(args.paths)

    processed_dir: Path = paths["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CFG] processed_dir = {processed_dir}")
    print(f"[CFG] filings_clean = {paths['filings_clean']}")
    print(f"[CFG] stopwords     = {paths['stopwords']}")

    stop = load_stopwords(paths["stopwords"])
    print(f"[LOAD] stopwords: {len(stop)} terms")

    filings_df = load_monthly_docs(paths["filings_clean"])  # <— ALWAYS from filings_clean

    # Split baseline vs rest
    seed_start = pd.Timestamp(args.seed_start)
    seed_end = pd.Timestamp(args.seed_end)
    print(f"[SEED] window: {seed_start.date()} → {seed_end.date()}")

    base = filings_df[(filings_df["date"] >= seed_start) & (filings_df["date"] <= seed_end)]
    print(f"[SEED] base docs: {len(base):,}")

    # 1) Baseline lexicon
    lex_path = processed_dir / "word_baseline_lexicon_2005_2010.json"
    seed_lex = build_baseline_lexicon(
        base_docs=base,
        stop=stop,
        min_ngr=args.min_ngr,
        max_ngr=args.max_ngr,
        min_df_frac=args.min_df_frac,
        max_df_frac=args.max_df_frac,
        out_path=lex_path,
    )

    # 2) Yearly snapshots
    seed_end_year = int(seed_end.year)
    lex_by_year = build_lexicon_snapshots(
        all_docs=filings_df,
        seed_lexicon=seed_lex,
        stop=stop,
        min_ngr=args.min_ngr,
        max_ngr=args.max_ngr,
        yearly_min_df_abs=args.yearly_min_df_abs,
        yearly_max_df_frac=args.yearly_max_df_frac,
        seed_end_year=seed_end_year,
    )

    # 3) Monthly panel (streaming, to avoid OOM)
    print("[PANEL] building monthly TF/DF (streaming) …")
    chunks = []
    for chunk in build_monthly_word_panel_streaming(
        monthly_docs=filings_df,
        stop=stop,
        min_ngr=args.min_ngr,
        max_ngr=args.max_ngr,
        seed_lexicon=seed_lex,
        lex_by_year=lex_by_year,
        flush_every="Q",
    ):
        chunks.append(chunk)
    panel = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["gram","date","tf","df_docs","month_tokens","month_docs"]).astype({"date":"datetime64[ns]"})

    # 4) Shares + momentum
    print("[PANEL] computing shares + momentum …")
    panel = add_shares_and_momentum(panel)

    # 5) Save parquet
    out_parquet = processed_dir / "word_trends_monthly.parquet"
    panel.to_parquet(out_parquet, index=False)
    print(f"[OK] wrote {out_parquet} with {len(panel):,} rows; span {panel['date'].min()} → {panel['date'].max()}")

    # 6) Optional summaries
    if args.export_summaries:
        export_summaries(panel, processed_dir / "summaries")
        print(f"[OK] summaries → {processed_dir / 'summaries'}")


if __name__ == "__main__":
    main()

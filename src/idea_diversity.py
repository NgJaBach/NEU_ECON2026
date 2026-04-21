"""
Idea Diversity Scorer
=====================
Computes semantic diversity for lists of ideas.

Methodology (based on DAT / Sentence-BERT literature):
  diversity(list) = mean pairwise cosine distance across all idea pairs

  diversity ∈ [0, 1]  where:
    0 = all ideas are identical / fully overlapping
    1 = all ideas are maximally different (orthogonal)

Embedding methods (tried in priority order):
  1. sentence-transformers multilingual  (semantic, best quality)
  2. TF-IDF word + char n-grams          (lexical, works immediately)

Input:  Excel / CSV file with an 'idea' column + a grouping column
Output: CSV + JSON with one diversity score per group
"""

import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

class TFIDFEmbedder:
    """
    TF-IDF embedding using word unigrams/bigrams + character 3-4-grams.
    Works for any language with whitespace-separated words (incl. Vietnamese).

    Vocabulary is fitted on ALL ideas globally so diversity scores are
    comparable across different groups.
    """

    def __init__(self):
        self.word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=1,
        )
        self._fitted = False
        self.method_name = "TF-IDF (word 1-2gram + char 3-5gram, global vocab)"

    def fit(self, all_ideas: list[str]) -> None:
        """Fit vocabulary on the entire corpus (call once before per-group scoring)."""
        corpus = [str(s).strip() for s in all_ideas if s and str(s).strip()]
        self.word_vec.fit(corpus)
        self.char_vec.fit(corpus)
        self._fitted = True
        log.info(
            f"TF-IDF fitted: {len(self.word_vec.vocabulary_)} word n-grams, "
            f"{len(self.char_vec.vocabulary_)} char n-grams"
        )

    def _normalise(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    def embed(self, corpus: list[str]) -> np.ndarray:
        """Transform a list of ideas to L2-normalised TF-IDF feature vectors."""
        if not self._fitted:
            # Fallback: fit on this corpus if global fit was not called
            self.fit(corpus)
        w = self.word_vec.transform(corpus).toarray()
        c = self.char_vec.transform(corpus).toarray()
        combined = np.hstack([w, c])
        return self._normalise(combined)


class SentenceBERTEmbedder:
    """
    Multilingual Sentence-BERT embeddings.
    Model: paraphrase-multilingual-MiniLM-L12-v2
      - Supports 50+ languages including Vietnamese
      - ~117 MB download; cached after first use
    """

    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer  # lazy import
        log.info(f"Loading Sentence-BERT model: {self.MODEL_NAME}")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.method_name = f"Sentence-BERT ({self.MODEL_NAME})"
        log.info("Model loaded.")

    def embed(self, corpus: list[str]) -> np.ndarray:
        vecs = self.model.encode(corpus, show_progress_bar=False, convert_to_numpy=True)
        # Normalise to unit sphere so cosine distance ∈ [0,1]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms


def get_embedder(prefer_sbert: bool = True):
    """Return best available embedder."""
    if prefer_sbert:
        try:
            emb = SentenceBERTEmbedder()
            log.info(f"Using: {emb.method_name}")
            return emb
        except ImportError:
            log.warning("sentence-transformers not installed — falling back to TF-IDF.")
        except Exception as exc:
            log.warning(f"Sentence-BERT failed ({exc}) — falling back to TF-IDF.")

    emb = TFIDFEmbedder()
    log.info(f"Using: {emb.method_name}")
    return emb


# ---------------------------------------------------------------------------
# Core diversity function
# ---------------------------------------------------------------------------

def average_pairwise_cosine_distance(ideas: list[str], embedder) -> tuple[float, dict]:
    """
    Diversity = mean cosine distance over all unique pairs of ideas.

    Normalised to [0,1]:
      score = mean(1 - cosine_similarity(i, j)) / 2
    The /2 maps the theoretical maximum distance of 2 (anti-parallel unit vectors)
    to 1.0, giving a consistent [0,1] range regardless of embedding method.

    Returns (diversity_score, stats_dict).
    """
    ideas = [str(s).strip() for s in ideas if s and str(s).strip()]
    if len(ideas) < 2:
        return 0.0, {"min": 0.0, "max": 0.0, "std": 0.0, "n_pairs": 0}

    vecs = embedder.embed(ideas)          # shape (N, D), L2-normalised
    dist_mat = cosine_distances(vecs)     # (N, N), values ∈ [0, 2]

    # Upper-triangle only (unique pairs)
    n = len(ideas)
    idx = np.triu_indices(n, k=1)
    pairwise = dist_mat[idx]              # shape (N*(N-1)/2,)

    raw_mean = float(np.mean(pairwise))
    raw_std  = float(np.std(pairwise))
    raw_min  = float(np.min(pairwise))
    raw_max  = float(np.max(pairwise))

    # Normalise to [0,1] via /2
    diversity = round(min(1.0, raw_mean / 2.0), 6)
    stats = {
        "min":    round(min(1.0, raw_min / 2.0), 6),
        "max":    round(min(1.0, raw_max / 2.0), 6),
        "std":    round(raw_std / 2.0, 6),
        "n_pairs": len(pairwise),
    }
    return diversity, stats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

IDEA_COL_CANDIDATES = ["idea", "ideas", "Idea", "Ideas", "text", "Text"]
GROUP_COL_CANDIDATES = [
    "bot_type", "condition", "group", "list", "category",
    "BotType", "Condition", "Group", "List", "Category",
]


def load_data(filepath: str | Path) -> pd.DataFrame:
    path = Path(filepath)
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    elif path.suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    log.info(f"Loaded {len(df)} rows from {path.name}")
    return df


def detect_columns(df: pd.DataFrame, idea_col: str | None, group_col: str | None):
    """Auto-detect idea and group columns if not specified."""
    cols = list(df.columns)

    if idea_col is None:
        for c in IDEA_COL_CANDIDATES:
            if c in cols:
                idea_col = c
                break
        if idea_col is None:
            raise ValueError(
                f"Cannot auto-detect idea column. Columns: {cols}\n"
                "Specify with --idea-col"
            )

    if group_col is None:
        for c in GROUP_COL_CANDIDATES:
            if c in cols:
                group_col = c
                break
        if group_col is None:
            log.warning("No group column found — treating all ideas as one list.")
            df = df.copy()
            df["_single_group"] = "all_ideas"
            group_col = "_single_group"

    log.info(f"Idea column: '{idea_col}' | Group column: '{group_col}'")
    return df, idea_col, group_col


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def score_diversity(
    filepath: str | Path,
    idea_col: str | None = None,
    group_col: str | None = None,
    row_filter_col: str | None = None,
    row_filter_val: str | None = None,
    use_sbert: bool = True,
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """
    Full pipeline: load → group → embed → score → save.

    Parameters
    ----------
    filepath       : Path to Excel/CSV file
    idea_col       : Column containing idea text (auto-detected if None)
    group_col      : Column to group ideas by (auto-detected if None)
    row_filter_col : Optional column to pre-filter rows (e.g. 'row_type')
    row_filter_val : Value to keep in row_filter_col (e.g. 'submission')
    use_sbert      : Try Sentence-BERT first (falls back to TF-IDF)
    output_dir     : Where to write results

    Returns
    -------
    DataFrame with columns: group, n_ideas, diversity_score, method
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(filepath)
    df, idea_col, group_col = detect_columns(df, idea_col, group_col)

    # Optional row filter
    if row_filter_col and row_filter_val:
        before = len(df)
        df = df[df[row_filter_col].astype(str) == row_filter_val]
        log.info(f"Filtered to {row_filter_col}=={row_filter_val!r}: {before} → {len(df)} rows")

    # Drop rows with empty ideas
    df = df[df[idea_col].notna() & (df[idea_col].astype(str).str.strip() != "")]
    log.info(f"Rows with non-empty ideas: {len(df)}")

    # Group ideas
    groups = df.groupby(group_col)[idea_col].apply(list).to_dict()
    log.info(f"Found {len(groups)} groups: {sorted(groups.keys())}")

    # Initialise embedder
    embedder = get_embedder(prefer_sbert=use_sbert)

    # For TF-IDF: fit global vocabulary on ALL ideas for cross-group comparability
    if isinstance(embedder, TFIDFEmbedder):
        all_ideas_flat = [idea for ideas in groups.values() for idea in ideas]
        all_ideas_flat = [str(s).strip() for s in all_ideas_flat if s and str(s).strip()]
        embedder.fit(all_ideas_flat)

    results = []
    for group_name, ideas in sorted(groups.items()):
        log.info(f"  Group '{group_name}': {len(ideas)} ideas → computing diversity...")
        score, stats = average_pairwise_cosine_distance(ideas, embedder)
        results.append(
            {
                "group": group_name,
                "n_ideas": len(ideas),
                "diversity_score": score,
                "pairwise_min": stats["min"],
                "pairwise_max": stats["max"],
                "pairwise_std": stats["std"],
                "n_pairs": stats["n_pairs"],
                "method": embedder.method_name,
            }
        )
        log.info(f"    → diversity = {score:.4f} (min={stats['min']:.3f}, max={stats['max']:.3f}, std={stats['std']:.3f})")

    result_df = pd.DataFrame(results)

    # ---- Save outputs ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_tag = "sbert" if "BERT" in embedder.method_name else "tfidf"
    stem = f"idea_diversity_{method_tag}_{ts}"

    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"

    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_at": ts,
                "method": embedder.method_name,
                "input_file": str(filepath),
                "idea_column": idea_col,
                "group_column": group_col,
                "n_groups": len(results),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log.info(f"\nResults saved to:\n  {csv_path}\n  {json_path}")

    # Print summary table
    print("\n" + "=" * 72)
    print("IDEA DIVERSITY RESULTS")
    print("=" * 72)
    print(f"{'Group':<20} {'N ideas':>8} {'Diversity':>12} {'Min':>8} {'Max':>8} {'Std':>8}")
    print("-" * 72)
    for row in results:
        print(
            f"{str(row['group']):<20} {row['n_ideas']:>8} "
            f"{row['diversity_score']:>12.4f} "
            f"{row['pairwise_min']:>8.4f} {row['pairwise_max']:>8.4f} {row['pairwise_std']:>8.4f}"
        )
    print("=" * 72)
    print(f"Method: {embedder.method_name}")
    print(f"Output: {csv_path}")
    print()

    return result_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute Idea Diversity scores for lists of ideas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/idea_diversity.py data/data_clean.xlsx
  python src/idea_diversity.py data/my_data.xlsx --group-col condition --row-filter-col row_type --row-filter-val submission
  python src/idea_diversity.py data/my_data.xlsx --no-sbert   # force TF-IDF
        """,
    )
    p.add_argument("filepath", help="Excel or CSV file with ideas")
    p.add_argument("--idea-col", default=None, help="Column with idea text (auto-detected)")
    p.add_argument("--group-col", default=None, help="Column to group ideas (auto-detected)")
    p.add_argument("--row-filter-col", default=None, help="Filter: column name")
    p.add_argument("--row-filter-val", default=None, help="Filter: value to keep")
    p.add_argument("--no-sbert", action="store_true", help="Skip Sentence-BERT, use TF-IDF")
    p.add_argument("--output-dir", default="results", help="Where to save results (default: results/)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score_diversity(
        filepath=args.filepath,
        idea_col=args.idea_col,
        group_col=args.group_col,
        row_filter_col=args.row_filter_col,
        row_filter_val=args.row_filter_val,
        use_sbert=not args.no_sbert,
        output_dir=args.output_dir,
    )

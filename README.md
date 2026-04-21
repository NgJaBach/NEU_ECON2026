# NEU ECON 2026 — Idea Diversity Scoring

Measures semantic diversity for lists of ideas from a human–AI co-creation experiment.
Output: one score per experimental condition, in **[0, 1]** (higher = more diverse ideas).

---

## Project Structure

```
.
├── data/               # Input data files (Excel / CSV)
│   └── data_clean.xlsx # Sample/reference dataset (English, 4 conditions)
├── docs/               # Methodology notes
│   └── idea_diversity_methodology.md
├── results/            # Auto-generated output (CSV + JSON per run)
├── src/
│   └── idea_diversity.py   # Main scoring script
├── requirements.txt
└── README.md
```

---

## Setup

**Python 3.10+ required.**

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** `pip install torch` installs the GPU version by default (~2 GB).
> For CPU-only (smaller, sufficient for this task):
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```
> Then run `pip install -r requirements.txt` as normal.

The first run will download the multilingual Sentence-BERT model (~117 MB, cached automatically).

---

## Usage

### Basic (auto-detects columns)
```bash
python src/idea_diversity.py data/data_clean.xlsx
```

### Full options
```bash
python src/idea_diversity.py data/my_data.xlsx \
  --group-col  bot_type \          # column that defines the 12 lists
  --row-filter-col  row_type \     # optional: filter rows first
  --row-filter-val  submission \   #   keep only rows where row_type == "submission"
  --output-dir  results
```

### Force TF-IDF (no internet / no GPU needed)
```bash
python src/idea_diversity.py data/my_data.xlsx --no-sbert
```

### All flags
| Flag | Default | Description |
|------|---------|-------------|
| `filepath` | — | Path to `.xlsx` or `.csv` |
| `--group-col` | auto-detected | Column that separates ideas into lists |
| `--idea-col` | auto-detected (`idea`) | Column containing idea text |
| `--row-filter-col` | none | Filter: column name |
| `--row-filter-val` | none | Filter: value to keep |
| `--no-sbert` | off | Use TF-IDF instead of Sentence-BERT |
| `--output-dir` | `results/` | Where to write output files |

---

## Input Format

| Column | Required | Description |
|--------|----------|-------------|
| `idea` | Yes | Text of the idea (Vietnamese or any language) |
| `bot_type` / `condition` / `group` | Yes | Which list this idea belongs to |
| `row_type` | Recommended | Use `--row-filter-val submission` to exclude incomplete rows |

---

## Output

Two files are written to `results/` after each run:

**`idea_diversity_sbert_YYYYMMDD_HHMMSS.csv`**
```
group,n_ideas,diversity_score,pairwise_min,pairwise_max,pairwise_std,n_pairs,method
control,97,0.2762,...
feedback,96,0.2908,...
```

**`idea_diversity_sbert_YYYYMMDD_HHMMSS.json`** — same data plus run metadata.

The key column is **`diversity_score`**: one number per list, in [0, 1].

---

## Method

**Average Pairwise Cosine Distance** — the standard approach from the DAT
(Divergent Association Task) and Sentence-BERT literature:

1. Embed each idea with `paraphrase-multilingual-MiniLM-L12-v2` (supports Vietnamese)
2. For every unique pair of ideas in a list: compute cosine distance
3. Average all pairwise distances → divide by 2 → diversity score in [0, 1]

If `sentence-transformers` is unavailable, falls back to TF-IDF with character + word n-grams.

See [`docs/idea_diversity_methodology.md`](docs/idea_diversity_methodology.md) for full details and references.

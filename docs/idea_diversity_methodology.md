# Idea Diversity Scoring — Methodology & Implementation Notes

**Author:** Claude (automated documentation)
**Date:** 2026-04-21
**Script:** `src/idea_diversity.py`

---

## 1. What This Measures

**Idea Diversity** quantifies how semantically different the ideas in a list are from one another.

- Output: a single number in **[0, 1]** per list/group
  - `0` → all ideas are essentially identical
  - `1` → ideas are maximally different (orthogonal in semantic space)
- Input: a list of idea texts (any language, including Vietnamese)

---

## 2. Theoretical Grounding

### Primary References
| Paper | Key Concept Used |
|-------|-----------------|
| Sentence-BERT (Reimers & Gurevych, 2019) | Multilingual semantic embeddings |
| DAT — Divergent Association Task (Olson et al., 2021, *PNAS*) | Average pairwise semantic distance = creativity measure |
| Human Creativity in the Age of LLMs | Semantic diversity as experimental outcome |
| Partnering with Generative AI (Experiment) | Condition-based diversity scoring |

### Core Formula

$$\text{Diversity}(L) = \frac{1}{|P|} \sum_{(i,j) \in P} \frac{d_\text{cosine}(e_i, e_j)}{2}$$

where:
- $L = \{t_1, \ldots, t_N\}$ is the list of $N$ ideas
- $P = \{(i,j) : i < j\}$ are all unique idea pairs ($|P| = N(N-1)/2$)
- $e_k$ is the L2-normalised embedding vector for idea $t_k$
- $d_\text{cosine}(u,v) = 1 - \cos(u,v) \in [0, 2]$ for unit vectors
- Dividing by **2** maps the range to $[0, 1]$

**Why average pairwise distance?**  
The DAT methodology establishes that average semantic distance between generated items reliably predicts creative ability. Extending this to idea lists: higher average pairwise distance = more diverse thinking.

---

## 3. Embedding Methods

Two methods are implemented, tried in priority order:

### Method 1: Sentence-BERT (Recommended)
- **Model:** `paraphrase-multilingual-MiniLM-L12-v2`
- **Source:** HuggingFace (auto-downloaded, ~117 MB, cached after first use)
- **Languages:** 50+ including Vietnamese, English, and others
- **Dimension:** 384
- **Why:** Captures semantic meaning — ideas with similar concepts are close, even with different wording

### Method 2: TF-IDF (Fallback)
- **Features:** Word unigrams/bigrams + character 3–5-grams
- **Vocabulary:** Fitted on **all ideas globally** (same feature space across groups)
- **Why fallback:** Works without any model download; captures lexical (surface-level) diversity
- **Limitation:** TF-IDF sees "drive" and "operate" as unrelated; Sentence-BERT knows they are semantically close

---

## 4. Data Format

The script expects an Excel (`.xlsx`) or CSV (`.csv`) file with at least:

| Column | Description |
|--------|-------------|
| `idea` | The idea text (Vietnamese or other language) |
| `bot_type` (or `condition`, `group`, etc.) | Column to group ideas into lists |

Optional filter columns (e.g., `row_type == "submission"`) can be specified via `--row-filter-col` / `--row-filter-val`.

**Sample file used for validation:** `data/data_clean.xlsx`
- 486 rows, 67 columns
- 389 idea-bearing submission rows
- 4 groups: `control`, `feedback`, `improvement`, `suggestion`

---

## 5. Results

### Sentence-BERT Results (2026-04-21)

| Group | N ideas | Diversity (0–1) | Min pair | Max pair | Std |
|-------|---------|-----------------|----------|----------|-----|
| control | 97 | **0.2762** | 0.0555 | 0.5082 | 0.0707 |
| feedback | 96 | **0.2908** | 0.0968 | 0.5254 | 0.0774 |
| improvement | 101 | **0.3574** | 0.0697 | 0.5731 | 0.0776 |
| suggestion | 95 | **0.2802** | 0.0797 | 0.5235 | 0.0713 |

Ranking: **improvement > feedback ≈ suggestion > control**

### Validation against Existing Scores in data_clean.xlsx

The sample file already contained per-idea diversity scores (`idea_div_condition`) from a prior analysis. Averaging those scores per group gives:

| Group | Existing (from file) | My SBERT | Rank agreement |
|-------|---------------------|----------|----------------|
| control | 0.308 | 0.276 | ✓ #4 |
| feedback | 0.349 | 0.291 | ✓ #2–3 |
| improvement | 0.470 | 0.357 | ✓ #1 |
| suggestion | 0.354 | 0.280 | ✓ #2–3 |

Rankings agree at positions 1 and 4 (the clear cases). Positions 2–3 are a near-tie in both approaches (difference < 1.5% in existing, < 4% in mine) so the order is expected to vary slightly across methods.

---

## 6. How to Run

### Prerequisites
- Python interpreter: `C:\Users\ngbac\miniconda3\python.exe` (has numpy, pandas, sklearn, torch, sentence-transformers)
- Working directory: project root (`NEU_ECON2026/`)

### Basic usage (auto-detects columns, uses Sentence-BERT)
```bash
C:\Users\ngbac\miniconda3\python.exe src/idea_diversity.py data/data_clean.xlsx
```

### With explicit options
```bash
C:\Users\ngbac\miniconda3\python.exe src/idea_diversity.py data/my_data.xlsx \
  --group-col bot_type \
  --row-filter-col row_type \
  --row-filter-val submission \
  --output-dir results
```

### Force TF-IDF (no internet / faster)
```bash
C:\Users\ngbac\miniconda3\python.exe src/idea_diversity.py data/my_data.xlsx --no-sbert
```

### For the researcher's Vietnamese data (12 groups)
Replace `--group-col bot_type` with whatever column encodes the 12 experimental conditions.

---

## 7. Output Files

All results are saved to `results/` with a timestamp:

| File | Content |
|------|---------|
| `idea_diversity_sbert_YYYYMMDD_HHMMSS.csv` | CSV table: group, n_ideas, diversity_score, min, max, std, n_pairs, method |
| `idea_diversity_sbert_YYYYMMDD_HHMMSS.json` | Full JSON with metadata + results |

The **`diversity_score`** column is the primary output for research: one number per group, in [0, 1].

---

## 8. Implementation Decisions

### Why divide by 2?
Sentence-BERT unit vectors can be anti-parallel (cosine similarity = −1), giving cosine distance = 2. Dividing by 2 maps the theoretical maximum to 1.0, ensuring consistent [0,1] range regardless of embedding method.

### Why global TF-IDF vocabulary?
Fitting TF-IDF separately per group would create incomparable feature spaces. Global fitting uses the same vocabulary for all groups, making the scores directly comparable.

### Why `paraphrase-multilingual-MiniLM-L12-v2`?
- Smallest multilingual model with strong Vietnamese performance
- 117 MB vs 278 MB for the larger `mpnet` variant
- Well-validated in cross-lingual semantic similarity benchmarks
- Explicitly cited in creativity research using Sentence-BERT embeddings

---

## 9. File Structure

```
NEU_ECON2026/
├── data/
│   └── data_clean.xlsx          ← sample dataset (English, used for validation)
├── docs/
│   └── idea_diversity_methodology.md  ← this file
├── results/
│   ├── idea_diversity_sbert_*.csv     ← Sentence-BERT results
│   ├── idea_diversity_sbert_*.json
│   ├── idea_diversity_tfidf_*.csv     ← TF-IDF results
│   └── idea_diversity_tfidf_*.json
└── src/
    └── idea_diversity.py              ← main scoring script
```

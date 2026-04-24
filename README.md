# NEU ECON 2026 — Idea Diversity Scoring

Đo lường **độ đa dạng ngữ nghĩa của ý tưởng** trong thực nghiệm human–AI co-creation.  
Đầu ra: một điểm số trong **[0, 1]** cho mỗi nhóm (cao hơn = ý tưởng đa dạng hơn).

Dữ liệu chính: `data/Data.xlsx` — 81 người tham gia, 3 điều kiện AI × 4 bài kiểm tra = **12 nhóm**.

---

## Cấu Trúc Project

```
.
├── data/
│   ├── Data.xlsx              # Dữ liệu chính (tiếng Việt, 81 người, wide format)
│   └── data_clean.xlsx        # Dữ liệu mẫu dùng để validation (tiếng Anh)
├── docs/
│   └── bao_cao_idea_diversity.md   # Báo cáo đầy đủ: thuật toán, kết quả, ANOVA
├── results/
│   ├── diversity_12groups_sbert_20260422_103416.csv  # Kết quả chính (12 nhóm)
│   ├── diversity_stats_20260424_144713.json          # ANOVA + Tukey HSD
│   └── diversity_stats_20260424_144713.csv           # Thống kê mô tả tổng hợp
├── src/
│   ├── idea_diversity.py      # Core pipeline (reusable, CLI)
│   ├── score_data.py          # Tính điểm 12 nhóm từ Data.xlsx
│   └── analyze_diversity.py   # Kiểm định thống kê (ANOVA, Tukey HSD)
├── papers/                    # Tài liệu tham khảo
├── requirements.txt
└── README.md
```

---

## Cài Đặt

**Yêu cầu:** Python 3.10+

```bash
pip install -r requirements.txt
```

> **Lưu ý:** `pip install torch` mặc định cài bản GPU (~2 GB).
> Để cài CPU-only (nhẹ hơn, đủ dùng cho task này):
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

Lần chạy đầu tiên sẽ tự tải model Sentence-BERT (~117 MB, tự động cache sau đó).

> **Môi trường đã test:** `C:\Users\ngbac\miniconda3\python.exe` (Python 3.13, conda base env)

---

## Cách Chạy

### Tính điểm 12 nhóm (kết quả chính)

```bash
python src/score_data.py
```

Đọc `data/Data.xlsx`, tính điểm đa dạng cho 12 nhóm (3 điều kiện × 4 bài), lưu vào `results/`.

### Kiểm định thống kê (ANOVA + Tukey HSD)

```bash
python src/analyze_diversity.py
```

Đọc CSV kết quả mới nhất trong `results/`, chạy one-way ANOVA, two-way ANOVA, và Tukey HSD post-hoc.

### Pipeline tổng quát (cho dữ liệu mới)

```bash
python src/idea_diversity.py data/my_data.xlsx \
  --group-col  condition \       # cột phân nhóm
  --row-filter-col  row_type \   # (tuỳ chọn) lọc dòng
  --row-filter-val  submission \
  --output-dir  results
```

#### Tất cả các tham số

| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `filepath` | — | Đường dẫn file `.xlsx` hoặc `.csv` |
| `--group-col` | tự nhận diện | Cột phân nhóm ý tưởng |
| `--idea-col` | tự nhận diện (`idea`) | Cột chứa nội dung ý tưởng |
| `--row-filter-col` | không | Tên cột lọc |
| `--row-filter-val` | không | Giá trị cần giữ lại |
| `--no-sbert` | tắt | Dùng TF-IDF thay Sentence-BERT |
| `--output-dir` | `results/` | Thư mục lưu kết quả |

---

## Định Dạng Đầu Vào

### Data.xlsx (wide format — 4 cột ý tưởng trên mỗi dòng)

| Cột | Mô tả |
|-----|-------|
| `Bạn được chia vào nhóm điều kiện nào` | Điều kiện: Nhóm A/B/C |
| `Hãy đề xuất ... "Camera "` | Ý tưởng bài Camera |
| `Hãy đề xuất ... "Sensor (...)"` | Ý tưởng bài Sensor |
| `Hãy đề xuất ... Đèn (Lights)` | Ý tưởng bài Đèn |
| `Hãy đề xuất ... "Loa (Speakers)"` | Ý tưởng bài Loa |

### Long format (dùng với `idea_diversity.py`)

| Cột | Bắt buộc | Mô tả |
|-----|----------|-------|
| `idea` | Có | Nội dung ý tưởng (tiếng Việt hoặc bất kỳ ngôn ngữ nào) |
| `condition` / `group` | Có | Nhóm của ý tưởng |
| `row_type` | Khuyến nghị | Dùng `--row-filter-val submission` để loại dòng không hợp lệ |

---

## Đầu Ra

### `diversity_12groups_sbert_YYYYMMDD_HHMMSS.csv` — Kết quả chính

```
group,condition,item,n_ideas,diversity_score,pairwise_min,pairwise_max,pairwise_std,n_pairs,method
A_Question × Camera,A_Question,Camera,26,0.209088,...
...
```

Cột quan trọng nhất: **`diversity_score`** — một số trong [0, 1] cho mỗi nhóm.

### `diversity_stats_YYYYMMDD_HHMMSS.json` — Thống kê kiểm định

Chứa: thống kê mô tả (mean, std, min, max) theo điều kiện và bài, kết quả ANOVA, bảng Tukey HSD.

---

## Phương Pháp

**Average Pairwise Cosine Distance** — từ DAT (Divergent Association Task, Olson et al. 2021) mở rộng sang sentence embeddings:

1. Nhúng mỗi ý tưởng bằng `paraphrase-multilingual-MiniLM-L12-v2` (hỗ trợ tiếng Việt)
2. Tính cosine distance cho mọi cặp ý tưởng trong nhóm
3. Lấy trung bình tất cả khoảng cách → chia 2 → điểm đa dạng ∈ [0, 1]

TF-IDF (word + char n-gram) là phương án dự phòng khi không có `sentence-transformers`.

Xem [`docs/bao_cao_idea_diversity.md`](docs/bao_cao_idea_diversity.md) để biết chi tiết thuật toán, kết quả, và kiểm định thống kê.

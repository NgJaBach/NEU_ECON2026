# Báo Cáo: Đo Lường Idea Diversity trong Nghiên Cứu Human–AI Co-Creation

**Ngày:** 2026-04-24  
**Script chính:** `src/score_data.py` · `src/analyze_diversity.py`  
**Dữ liệu:** `data/Data.xlsx` (81 người tham gia, 3 điều kiện × 4 bài kiểm tra = 12 nhóm)

---

## 1. Mục Tiêu

Nghiên cứu đặt câu hỏi: **AI tương tác theo các cách khác nhau (đặt câu hỏi, gợi ý, hoặc không can thiệp) có tác động đến sự đa dạng ý tưởng của con người không?**

Chỉ số **Idea Diversity** (Đa dạng ý tưởng) được tính cho từng nhóm thực nghiệm, cho ra **một con số trong [0, 1]**:

| Giá trị | Ý nghĩa |
|---------|---------|
| **0.0** | Tất cả ý tưởng gần như giống nhau (trùng lặp về ngữ nghĩa) |
| **1.0** | Các ý tưởng hoàn toàn khác nhau (phân kỳ tối đa) |

---

## 2. Thiết Kế Thực Nghiệm

| Điều kiện | Mô tả | Ký hiệu | n |
|-----------|-------|---------|---|
| Nhóm A – Question mode | AI đặt câu hỏi kích thích tư duy | A_Question | 26 |
| Nhóm B – Suggestion mode | AI đưa ra gợi ý trực tiếp | B_Suggestion | 26 |
| Nhóm C – Control | Chỉ con người, không có AI | C_Control | 29 |

Mỗi người tham gia trả lời **4 bài kiểm tra** (4 đồ vật cần nghĩ ra công dụng mới):

| Bài | Đồ vật |
|-----|--------|
| Rr1 | Camera |
| Rr2 | Sensor (cảm biến) |
| Rr3 | Đèn (Lights) |
| Rr4 | Loa (Speakers) |

→ **12 nhóm** = 3 điều kiện × 4 bài

---

## 3. Phương Pháp Đo Lường (Phương Pháp Chính)

### 3.1. Nền Tảng Lý Thuyết

Phương pháp này dựa trực tiếp trên **Divergent Association Task (DAT)** (Olson et al., 2021, *PNAS*):

> *"Khoảng cách ngữ nghĩa trung bình giữa các từ/ý tưởng do một người tạo ra là thước đo đáng tin cậy của tư duy phân kỳ và khả năng sáng tạo."*

DAT đo khoảng cách ngữ nghĩa bằng word embeddings; chúng tôi mở rộng sang **sentence embeddings** (Sentence-BERT) để xử lý ý tưởng dạng câu thay vì từ đơn, và áp dụng ở **cấp độ nhóm** thay vì cá nhân.

### 3.2. Công Thức

$$\text{IdeaDiversity}(G) = \frac{1}{|P|} \sum_{(i,j) \in P} \frac{d_{\cos}(\mathbf{e}_i, \mathbf{e}_j)}{2}$$

Trong đó:
- $G = \{t_1, t_2, \ldots, t_N\}$: tập hợp $N$ ý tưởng của một nhóm
- $P = \{(i,j) : i < j\}$: tất cả $\frac{N(N-1)}{2}$ cặp ý tưởng duy nhất
- $\mathbf{e}_k$: vector embedding L2-chuẩn hóa của ý tưởng $t_k$ (384 chiều)
- $d_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - \cos(\mathbf{u}, \mathbf{v}) \in [0, 2]$ (khoảng cách cosine)
- **Chia 2** để chuẩn hóa về $[0, 1]$ (vì với vector đơn vị, khoảng cách cosine tối đa = 2)

**Trực giác:** Lấy trung bình khoảng cách của *mọi cặp ý tưởng* trong nhóm. Nhóm nào càng nhiều ý tưởng khác biệt nhau về mặt ngữ nghĩa → điểm càng cao.

### 3.3. Mô Hình Ngôn Ngữ: Sentence-BERT Đa Ngôn Ngữ

**Model:** `paraphrase-multilingual-MiniLM-L12-v2`  
**Nguồn:** [Reimers & Gurevych, 2019 – *Sentence-BERT*](https://arxiv.org/abs/1908.10084)

| Đặc điểm | Chi tiết |
|----------|---------|
| Kiến trúc | MiniLM-L12 với Transformer encoder |
| Kích thước | ~117 MB |
| Số ngôn ngữ | 50+ (bao gồm tiếng Việt) |
| Chiều embedding | 384 |
| Huấn luyện | Paraphrase pairs đa ngôn ngữ |

**Tại sao dùng Sentence-BERT?**

Không giống TF-IDF chỉ so sánh chữ bề mặt, SBERT hiểu **ngữ nghĩa**: hai câu "thiết bị chiếu sáng" và "đèn để rọi sáng" sẽ được nhận biết là giống nhau. Điều này phù hợp với mục tiêu nghiên cứu: đo sự đa dạng về *ý nghĩa* của ý tưởng, không chỉ về từ ngữ.

---

## 4. Kết Quả

### 4.1. Bảng 12 Nhóm (N ý tưởng, Điểm Đa Dạng)

| Nhóm | N | Điểm Đa Dạng | Min cặp | Max cặp | Std |
|------|---|:------------:|---------|---------|-----|
| **A_Question × Camera** | 26 | 0.2091 | 0.0410 | 0.4182 | 0.0702 |
| **A_Question × Sensor** | 26 | 0.2439 | 0.0784 | 0.4122 | 0.0635 |
| **A_Question × Đèn** | 26 | 0.1941 | 0.0373 | 0.4161 | 0.0714 |
| **A_Question × Loa** | 26 | **0.3162** | 0.1141 | 0.5071 | 0.0757 |
| **B_Suggestion × Camera** | 26 | 0.1962 | 0.0440 | 0.3032 | 0.0477 |
| **B_Suggestion × Sensor** | 26 | 0.2152 | 0.0558 | 0.4092 | 0.0664 |
| **B_Suggestion × Đèn** | 26 | 0.1738 | 0.0405 | 0.3315 | 0.0545 |
| **B_Suggestion × Loa** | 26 | 0.2970 | 0.1028 | 0.5106 | 0.0872 |
| **C_Control × Camera** | 29 | 0.2022 | 0.0654 | 0.3727 | 0.0594 |
| **C_Control × Sensor** | 29 | 0.2392 | 0.0538 | 0.4305 | 0.0717 |
| **C_Control × Đèn** | 29 | 0.1934 | 0.0689 | 0.3367 | 0.0503 |
| **C_Control × Loa** | 29 | 0.2961 | 0.0954 | 0.4795 | 0.0836 |

### 4.2. Trung Bình theo Điều Kiện

| Điều kiện | Mean | Std | Min | Max |
|-----------|:----:|-----|-----|-----|
| **A_Question** | **0.2408** | 0.0544 | 0.1941 | 0.3162 |
| B_Suggestion | 0.2205 | 0.0537 | 0.1738 | 0.2970 |
| C_Control | 0.2327 | 0.0467 | 0.1934 | 0.2961 |

**Thứ tự:** A_Question > C_Control > B_Suggestion

### 4.3. Trung Bình theo Bài Kiểm Tra

| Bài kiểm tra | Mean | Std | Min | Max |
|-------------|:----:|-----|-----|-----|
| **Loa** | **0.3031** | 0.0114 | 0.2961 | 0.3162 |
| Sensor | 0.2328 | 0.0154 | 0.2152 | 0.2439 |
| Camera | 0.2025 | 0.0065 | 0.1962 | 0.2091 |
| Đèn | 0.1871 | 0.0115 | 0.1738 | 0.1941 |

**Thứ tự:** Loa > Sensor > Camera > Đèn

---

## 5. Kiểm Định Thống Kê

### 5.1. One-Way ANOVA theo Điều Kiện

Kiểm định: *Điều kiện thực nghiệm có ảnh hưởng đến độ đa dạng ý tưởng không?*

| Chỉ số | Giá trị |
|--------|---------|
| F(2, 9) | 0.156 |
| p-value | 0.858 |
| η² (eta-squared) | 0.034 |
| Kruskal-Wallis H | 0.500 |
| p (Kruskal-Wallis) | 0.779 |

**Kết luận:** Không có sự khác biệt có ý nghĩa thống kê giữa 3 điều kiện khi xem xét riêng lẻ (p = 0.858). Cỡ mẫu n=4 điểm mỗi nhóm dẫn đến statistical power thấp.

### 5.2. One-Way ANOVA theo Bài Kiểm Tra

Kiểm định: *Loại đồ vật có ảnh hưởng đến độ đa dạng ý tưởng không?*

| Chỉ số | Giá trị |
|--------|---------|
| F(3, 8) | 58.63 |
| **p-value** | **< 0.001** |
| **η²** | **0.9565** |
| Kruskal-Wallis H | 10.38 |
| p (Kruskal-Wallis) | 0.016 |

**Kết luận:** Loại đồ vật có tác động RẤT LỚN đến độ đa dạng ý tưởng (p < 0.001, η² = 0.96 — effect size khổng lồ). Điều này có nghĩa là bản thân từng đồ vật kích thích sự phân kỳ ý tưởng ở mức độ khác nhau, bất kể điều kiện AI.

### 5.3. Two-Way ANOVA (Mô Hình Cộng Tính): Điều Kiện + Bài

Kiểm định: *Khi kiểm soát ảnh hưởng của bài kiểm tra, điều kiện có quan trọng không?*

Model: `diversity_score ~ C(condition) + C(item)`

| Nguồn biến thiên | SS | df | F | p-value |
|-----------------|----|----|---|---------|
| **Condition** | 0.0008 | 2 | **10.14** | **0.012** |
| **Item** | 0.0238 | 3 | **192.68** | **< 0.001** |
| Residual | 0.0002 | 6 | — | — |

**R² = 0.990 · Adj-R² = 0.982**

**Kết luận quan trọng:** Khi kiểm soát ảnh hưởng của bài kiểm tra (item), hiệu ứng điều kiện trở nên **có ý nghĩa thống kê (p = 0.012)**. Điều này cho thấy AI *có* ảnh hưởng đến độ đa dạng ý tưởng, nhưng hiệu ứng này bị che khuất bởi sự khác biệt lớn hơn nhiều giữa các bài kiểm tra.

### 5.4. Tukey HSD Post-hoc — Điều Kiện

| Cặp so sánh | Hiệu trung bình | p (adj) | Kết luận |
|-------------|:---------------:|:-------:|---------|
| A_Question vs B_Suggestion | −0.0203 | 0.846 | Không có ý nghĩa |
| A_Question vs C_Control | −0.0081 | 0.973 | Không có ý nghĩa |
| B_Suggestion vs C_Control | +0.0122 | 0.941 | Không có ý nghĩa |

### 5.5. Tukey HSD Post-hoc — Bài Kiểm Tra

| Cặp so sánh | Hiệu trung bình | p (adj) | Kết luận |
|-------------|:---------------:|:-------:|---------|
| **Loa vs Camera** | +0.1006 | < 0.001 | **Có ý nghĩa** |
| **Loa vs Sensor** | −0.0703 | 0.0003 | **Có ý nghĩa** |
| **Loa vs Đèn** | −0.1160 | < 0.001 | **Có ý nghĩa** |
| **Sensor vs Đèn** | −0.0457 | 0.006 | **Có ý nghĩa** |
| Camera vs Sensor | +0.0303 | 0.051 | Không có ý nghĩa (borderline) |
| Camera vs Đèn | −0.0154 | 0.421 | Không có ý nghĩa |

**Loa khác biệt đáng kể so với tất cả các bài còn lại.** Sensor khác biệt so với Đèn. Camera và Đèn không khác biệt nhau một cách thống kê.

---

## 6. Tóm Tắt Kết Quả

```
Điều kiện:  A_Question (0.2408) > C_Control (0.2327) > B_Suggestion (0.2205)
            → Không có ý nghĩa thống kê trong one-way ANOVA (n=4/nhóm)
            → Có ý nghĩa khi kiểm soát item (two-way ANOVA, p = 0.012)

Bài kiểm:  Loa (0.3031) >> Sensor (0.2328) > Camera (0.2025) ≈ Đèn (0.1871)
            → Có ý nghĩa cao (F = 58.6, p < 0.001, η² = 0.96)
            → Loa tạo ra ý tưởng đa dạng nhất; Đèn ít đa dạng nhất
```

**Diễn giải:** Item (loại đồ vật) là yếu tố chi phối mạnh nhất độ đa dạng ý tưởng. Khi kiểm soát item, điều kiện AI có tác động có ý nghĩa thống kê, nhưng cỡ mẫu thực nghiệm (n=4 điểm/nhóm trong one-way test) giới hạn statistical power. Nhóm A_Question (AI đặt câu hỏi) cho độ đa dạng cao nhất — phù hợp với giả thuyết rằng câu hỏi kích thích tư duy phân kỳ hơn gợi ý.

---

## 7. Pipeline Kỹ Thuật

```
Data.xlsx (81 người × 4 câu hỏi)
    ↓
score_data.py: phân nhóm 3 điều kiện × 4 bài = 12 nhóm
    ↓
SentenceBERTEmbedder.embed(ideas): mỗi ý tưởng → vector 384 chiều
    ↓
cosine_distances(vecs): ma trận N×N
    ↓
np.triu_indices(n, k=1): lấy N(N-1)/2 giá trị tam giác trên
    ↓
mean(pairwise) / 2: điểm đa dạng ∈ [0, 1]
    ↓
diversity_12groups_sbert_TIMESTAMP.csv (12 dòng)
    ↓
analyze_diversity.py: ANOVA + Tukey HSD          plot_diversity.py: box plot
    ↓                                                      ↓
diversity_stats_TIMESTAMP.json / .csv          figures/diversity_boxplot.png
```

### Các file đầu ra

| File | Mô tả |
|------|-------|
| `results/diversity_12groups_sbert_*.csv` | 12 điểm đa dạng (kết quả chính) |
| `results/diversity_stats_*.json` | ANOVA + Tukey HSD đầy đủ |
| `results/diversity_stats_*.csv` | Thống kê mô tả tổng hợp |
| `results/figures/diversity_boxplot.png` | Box plot pairwise distances theo điều kiện và vòng |

#### Giải thích cột trong `diversity_stats_*.csv`

File này gồm **7 hàng**: 3 hàng cho điều kiện (condition) và 4 hàng cho bài kiểm tra (item).

| Cột | Kiểu giá trị | Ý nghĩa |
|-----|-------------|---------|
| `factor` | `"condition"` hoặc `"item"` | Cho biết hàng này tổng hợp theo điều kiện AI hay theo loại đồ vật |
| `condition` | `A_Question`, `B_Suggestion`, `C_Control`, *(trống)* | Tên điều kiện — chỉ có giá trị ở các hàng `factor = condition` |
| `item` | `Camera`, `Sensor`, `Đèn`, `Loa`, *(trống)* | Tên bài kiểm tra — chỉ có giá trị ở các hàng `factor = item` |
| `n` | Số nguyên | Số nhóm được tổng hợp (condition: n=4 bài; item: n=3 điều kiện) |
| `mean` | [0, 1] | **Trung bình điểm đa dạng** của các nhóm trong nhóm tổng hợp này |
| `std` | ≥ 0 | Độ lệch chuẩn của điểm đa dạng — đo mức độ biến động giữa các nhóm |
| `min` | [0, 1] | Điểm đa dạng thấp nhất trong nhóm tổng hợp |
| `max` | [0, 1] | Điểm đa dạng cao nhất trong nhóm tổng hợp |

**Ví dụ đọc bảng:**

- Hàng `condition / A_Question / n=4 / mean=0.2408`: nhóm AI đặt câu hỏi có trung bình điểm đa dạng = 0.241, tính trên 4 bài kiểm tra (Camera, Sensor, Đèn, Loa).
- Hàng `item / Loa / n=3 / mean=0.3031`: bài Loa có trung bình điểm đa dạng = 0.303, tính trên 3 điều kiện (A, B, C). Std = 0.011 rất nhỏ → kết quả nhất quán giữa các điều kiện.
- Hàng `item / Đèn / n=3 / mean=0.1871`: bài Đèn có độ đa dạng thấp nhất, bất kể điều kiện AI nào.

---

## 8. Cách Chạy Lại

```bash
# Tính điểm đa dạng 12 nhóm
C:\Users\ngbac\miniconda3\python.exe src/score_data.py

# Chạy phân tích thống kê (ANOVA + Tukey HSD)
C:\Users\ngbac\miniconda3\python.exe src/analyze_diversity.py
```

---

## Phụ Lục A: Tại Sao Không Dùng Phương Pháp Khác

### A.1. TF-IDF (So Sánh Từ Vựng)

TF-IDF biến văn bản thành vector đặc trưng dựa trên tần suất xuất hiện của từ. Hai câu có cùng ý nghĩa nhưng từ khác nhau sẽ bị đánh giá là "khác biệt" (false positive cho diversity).

**Vấn đề:** Tiếng Việt có nhiều từ đồng nghĩa và cách diễn đạt đa dạng → TF-IDF đánh giá quá cao diversity thực sự.

Implemented trong `TFIDFEmbedder` (src/idea_diversity.py) làm fallback khi SBERT không khả dụng. **Không dùng làm kết quả chính thức.**

### A.2. BERTScore / ROUGE / BLEU

Các chỉ số này thiết kế cho bài toán so sánh bản dịch/tóm tắt (1 output vs 1 reference). Không phù hợp cho bài toán đo diversity của một tập hợp ý tưởng.

### A.3. Một Số Chỉ Số Originality Khác

Xem tài liệu: *Originality Score* (papers/), *Consensual Assessment Technique* (Amabile, 1982) — đây là đánh giá bởi chuyên gia (human rater), đòi hỏi nhiều người đánh giá và không tự động hoá được ở quy mô lớn.

---

## Phụ Lục B: Lưu Ý Thống Kê

**Giới hạn statistical power:**  
- One-way ANOVA theo điều kiện: n=4 điểm/nhóm (4 bài × 3 điều kiện chỉ cho 4 giá trị/nhóm) — cần ít nhất n=15–20 để có đủ power cho small-medium effects.
- Two-way ANOVA có thêm df từ việc kiểm soát item factor, giúp cải thiện power.

**Gợi ý cho phân tích nâng cao:**  
Nếu có thêm dữ liệu (nhiều điều kiện hơn hoặc nhiều bài hơn), nên dùng **mixed-effects model** với `participant` là random effect để xử lý repeated measures.

---

*Tài liệu này thay thế `docs/idea_diversity_methodology.md` (phiên bản cũ chỉ có kết quả mẫu).*

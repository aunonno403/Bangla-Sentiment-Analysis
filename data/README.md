# Dataset Documentation

## Overview

This directory contains the Bangla sentiment analysis dataset used for model training and evaluation.

- **`raw/`** — Original, untouched datasets from source
- **`processed/`** — Cleaned, tokenized, and preprocessed data ready for ML

---

## Data Composition

| Source | Size | Labels | Notes |
|--------|------|--------|-------|
| **SemEval-2024 Bangla** | ~500 | Pos, Neg, Neutral | Public benchmark from competition |
| **Custom-Labeled** | ~100-200 | Pos, Neg, Neutral | Self-collected from web sources |
| **Total** | ~600-700 | 3 classes | Mixed dataset |

---

## Data Collection & Labeling Guide

### Phase 1: Download Public Benchmark

**Option A: SemEval-2024 Bangla** (RECOMMENDED)
```bash
# Clone or download from:
# https://github.com/SemEval-2024/SemEval-2024-Task5-Bangla

# Save to data/raw/semeval_bangla.csv
```

**Option B: BanglaSentiment**
```bash
# Clone from:
# https://github.com/kaushikjadhav01/BanglaSentiment

# Save to data/raw/bangla_sentiment.csv
```

### Phase 2: Collect & Label Custom Data (100-200 examples)

**Label Definitions**:
- **Positive (1)**: Supportive, praising, expressing satisfaction or joy
  - Examples: "এটি দুর্দান্ত!", "আমি খুবই খুশি"
- **Negative (-1)**: Critical, complaining, expressing dissatisfaction or anger
  - Examples: "এটি খুবই খারাপ", "আমি হতাশ"
- **Neutral (0)**: Objective, factual, or mixed sentiment
  - Examples: "আজ মঙ্গলবার", "কিছু লোক এটি পছন্দ করে"

**Data Sources** (choose 1-2):
1. **YouTube Comments**: Bangla music/news channels → comments section
2. **News Site Comments**: Prothom Alo, BBC Bangla → user comments
3. **Social Media**: Twitter/X Bangla tweets, Facebook Bangla posts
4. **Product Reviews**: E-commerce sites (Daraz, etc.)

**Collection Process**:

1. Open a text editor or spreadsheet (Google Sheets / Excel)
2. Create columns: `text`, `sentiment`, `source`
3. Copy 10-15 relevant Bangla comments/texts
4. Label each with sentiment (positive/negative/neutral)
5. Record source (e.g., "YouTube_channel_name", "News_site")
6. Repeat until you have 100-200 examples
7. Save as **`data/raw/custom_labeled_data.csv`**

**CSV Format**:
```csv
text,sentiment,source
এটি অসাধারণ সেবা,positive,youtube_comment
খুবই হতাশাজনক,negative,news_comment
আজ সুন্দর দিন,neutral,twitter
```

**Estimated Time**: ~1.5 hours for 100-200 examples

**Tips**:
- Aim for balanced classes (~30-40% positive, ~30-40% negative, ~20-40% neutral)
- Avoid duplicates
- Include diverse Bangla dialects/writing styles if possible
- Document your labeling rationale (helps with model interpretability later)

---

## Data Schema

### Raw Data (CSV format)

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Bangla text sample |
| `sentiment` | string/int | Label: "positive"/"negative"/"neutral" (or 1/-1/0) |
| `source` | string | Data source (e.g., "youtube", "news", "twitter") |

### Processed Data (after preprocessing)

| Column | Type | Description |
|--------|------|-------------|
| `original_text` | string | Original Bangla text |
| `cleaned_text` | string | Normalized, lowercased text |
| `tokens` | list/string | Tokenized words |
| `sentiment` | int | 1 (positive), 0 (neutral), -1 (negative) |
| `source` | string | Original source |

---

## Data Quality Checks

Run these checks in `notebooks/01_eda.ipynb`:

- [ ] No missing values in `text` or `sentiment` columns
- [ ] Sentiment labels are consistent (no typos: "positive" vs "Positive")
- [ ] Text samples are valid UTF-8 Bangla (no encoding issues)
- [ ] Class distribution is reasonably balanced (no >80% single class)
- [ ] No duplicate texts
- [ ] Minimum text length >= 3 characters

---

## Dataset Split

For model training:
- **Training Set**: 80% (480-560 examples)
- **Test Set**: 20% (120-140 examples)

Split will be done in `src/model_training.py` with `train_test_split(random_state=42)` for reproducibility.

---

## Attribution & Licensing

- **SemEval-2024 Bangla**: Public research dataset, cite if used in publications
- **Custom Data**: Collected by you, free to use in this project
- **License**: MIT (see ../LICENSE)

---

## Next Steps

1. ✅ Download SemEval-2024 Bangla or BanglaSentiment → save to `data/raw/`
2. ✅ Collect & label 100-200 custom examples → save to `data/raw/custom_labeled_data.csv`
3. → Run `notebooks/01_eda.ipynb` to combine, validate, and analyze
4. → Save processed data to `data/processed/`

---

**Last Updated**: May 7, 2026 | **Phase**: 1 (Data Collection)

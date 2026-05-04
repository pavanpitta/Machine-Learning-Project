# Methodology: Step-by-step pipeline for each Research Question

This document describes the complete computational pipeline that takes the raw Spotify Tracks CSV and produces the final figure (PDF) and table (CSV) for each of the 7 Research Questions.

All seven notebooks share **Steps 1–4** (loading, filtering, target definition, feature engineering). They diverge at **Step 5** based on what each RQ asks.

---

## Shared steps (all notebooks)

### Step 1 — Load raw dataset
```python
df = pd.read_csv("tracks.csv", low_memory=False)
# Shape: (899,702, 16)
```

### Step 2 — Filter to modeling subset
```python
m = df.dropna(subset=["popularity","tempo","energy","danceability"]).copy()
# Shape: (~899,200, 16)
```

### Step 3 — Define binary target
```python
m["popular"] = (m["popularity"] >= 50).astype(int)
# Class balance: ~89% non-popular, ~11% popular
```

### Step 4 — Engineer 22 features

**Numeric audio features (12):**
- `tempo` (BPM)
- `energy`, `danceability`, `valence`, `acousticness`, `liveness`, `instrumentalness`, `speechiness` (all 0–1)
- `key` (0–11), `mode` (binary), `time_signature`
- `loudness_proxy = energy * (1 - acousticness)` (engineered interaction)

**Genre family dummies (5):**
- `genre_pop`, `genre_rock`, `genre_hiphop`, `genre_electronic`, `genre_other`
- Derived from substring matches inside the `genres` list field.

**Engineered audio combinations (5):**
- `is_high_energy` (energy > 0.7)
- `is_danceable` (danceability > 0.7)
- `is_acoustic` (acousticness > 0.5)
- `is_instrumental` (instrumentalness > 0.5)
- `valence_x_energy = valence * energy`

---

## RQ1 — Cross-Validation vs Single Hold-Out Stability

### Step 5 — Configure evaluation strategies
- Single hold-out: stratified 80/20 split, `random_state=42`.
- K-fold: Stratified 5-fold CV, `random_state=42`.

### Step 6 — Train 5 classifiers under both strategies
- Each model evaluated under both hold-out and 5-fold CV.
- Report mean ± std for CV metrics.

### Step 7 — Compute metrics
- For each model × strategy: Accuracy, F1, ROC-AUC.
- Save results table → `table_rq1_cv_stability.csv`.

### Step 8 — Visualise
- Side-by-side grouped bar chart comparing hold-out vs CV for each model.
- Error bars show CV standard deviation.
- Save → `fig_rq1_cv_stability.pdf` and `.png`.

---

## RQ2 — Permutation Importance vs Gain-Based Importance

### Step 5 — Train XGBoost (or GBM fallback)
- Same train/test split as RQ1 hold-out.

### Step 6 — Extract both importance types
- Gain-based: `model.feature_importances_`.
- Permutation-based: `sklearn.inspection.permutation_importance` on the test set, 30 repeats.

### Step 7 — Compare rankings
- Rank features by each method.
- Compute rank correlation (Spearman's ρ).
- Save → `table_rq2_permutation_importance.csv`.

### Step 8 — Visualise
- Two horizontal bar charts side by side (gain vs permutation), same feature order on y-axis.
- Save → `fig_rq2_permutation_importance.pdf`.

---

## RQ3 — Popularity Trends by Tempo Tier

### Step 5 — Bin tracks into tempo tiers
- Slow (< 90 BPM), Medium (90–120 BPM), Fast (120–150 BPM), Very Fast (> 150 BPM).

### Step 6 — Train one global model
- Same XGBoost/GBM as RQ2.

### Step 7 — Evaluate per tempo tier on the test set
- Subset the test set by tempo tier and compute per-tier popularity rate and model metrics.
- Save → `table_rq3_tempo_tier_popularity.csv`.

### Step 8 — Visualise (two-panel figure)
- Panel (a): Popularity rate per tempo tier as a bar chart.
- Panel (b): F1-Score and ROC-AUC per tier as grouped bars.
- Save → `fig_rq3_tempo_tier_popularity.pdf`.

---

## RQ4 — Genre Family Predictability

### Step 5 — Assign each track a primary genre family
- Pop, Rock, Hip-Hop, Electronic, Other (from `genres` substring matches).

### Step 6 — Train one global model
- Same setup as RQ2/RQ3.

### Step 7 — Evaluate per genre family
- Compute per-family popularity rate and model performance.
- Save → `table_rq4_genre_family.csv`.

### Step 8 — Visualise (two-panel figure)
- Panel (a): Popularity rate and dataset size per genre family.
- Panel (b): Accuracy, F1, AUC grouped bars per genre family.
- Save → `fig_rq4_genre_family.pdf`.

---

## RQ5 — Energy × Danceability Interaction Effects

### Step 5 — Bin `energy` and `danceability`
- Energy bins: Low (<0.4), Mid (0.4–0.7), High (>0.7).
- Danceability bins: Low (<0.4), Mid (0.4–0.7), High (>0.7).

### Step 6 — Train one global model
- Same setup as prior RQs.

### Step 7 — Compute per-cell metrics on the test set
- 3×3 interaction grid: popularity rate and F1 per (energy_bin × danceability_bin) cell.
- Save → `table_rq5_energy_danceability_interaction.csv`.

### Step 8 — Visualise (heatmap)
- Popularity rate heatmap (3×3) with annotated cell values.
- F1 heatmap alongside for comparison.
- Save → `fig_rq5_energy_danceability_interaction.pdf`.

---

## RQ6 — Class Imbalance Handling Strategies

### Step 5 — Apply four imbalance strategies
- Baseline: no resampling.
- Oversampling: SMOTE on training data.
- Undersampling: random undersampling of majority class.
- Class weights: `scale_pos_weight` (XGBoost) or `class_weight="balanced"`.

### Step 6 — Train XGBoost/GBM under each strategy
- Same train/test split throughout; resampling only applied to training data.

### Step 7 — Compute metrics
- Compare Accuracy, Precision, Recall, F1, ROC-AUC across all four strategies.
- Save → `table_rq6_imbalance_strategies.csv`.

### Step 8 — Visualise
- Grouped bar chart with four strategy groups × five coloured metric bars.
- Save → `fig_rq6_imbalance_strategies.pdf`.

---

## RQ7 — Acousticness Tier Analysis

### Step 5 — Assign acousticness tiers
- Electronic (acousticness < 0.2), Mixed (0.2–0.6), Acoustic (> 0.6).

### Step 6 — Train one global model
- Same XGBoost/GBM as prior RQs.

### Step 7 — Evaluate per acousticness tier
- Save → `table_rq7_acousticness_tiers.csv`.

### Step 8 — Visualise (two-panel figure)
- Panel (a): Popularity rate per tier + dataset size on twin axis.
- Panel (b): F1 and ROC-AUC grouped bars per tier.
- Save → `fig_rq7_acousticness_tiers.pdf`.

---

## Verification

Each notebook also writes a PNG version of its figure for quick preview. The CSV tables are formatted for direct inclusion in a LaTeX or Word report.

All seven notebooks are independent — running RQ4 does not require running RQ1 first. Each notebook is self-contained and reproducible with `random_state=42`.

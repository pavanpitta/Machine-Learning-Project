# Predicting Spotify Track Popularity from Audio Features

> A supervised binary classification project using the Spotify Tracks Dataset.
> Submission 2 — Machine Learning assignment.

This repository contains the proposal, code, and outputs for predicting whether a Spotify track will be **popular** (popularity score ≥ 50) using audio features such as tempo, energy, danceability, valence, acousticness, instrumentalness, and genre information.

---

## Project at a glance

| Field | Value |
|---|---|
| **Task** | Binary Classification (Supervised Learning) |
| **Target** | `popular = 1 if popularity >= 50, else 0` |
| **Dataset** | Spotify Tracks Dataset (`tracks.csv`) |
| **Raw size** | 899,702 rows × 16 columns |
| **Modeling subset** | ~899,200 rows × 22 engineered features |
| **Class balance** | ~89% non-popular / ~11% popular |
| **Best F1-Score** | XGBoost (F1 = 0.547, Acc = 0.892) |
| **Best ROC-AUC** | Random Forest (AUC = 0.851) |
| **Research questions** | 7 |

---

## Repository structure

```
.
├── Proposal.docx            # Word version of the proposal
├── Proposal.pdf             # PDF version of the proposal
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── notebooks/
│   ├── RQ1.ipynb            # Cross-validation vs single hold-out stability
│   ├── RQ2.ipynb            # Permutation importance vs gain-based importance
│   ├── RQ3.ipynb            # Popularity by tempo tier
│   ├── RQ4.ipynb            # Genre-family predictability
│   ├── RQ5.ipynb            # Energy x Danceability interaction effects
│   ├── RQ6.ipynb            # Class imbalance handling strategies
│   └── RQ7.ipynb            # Acousticness tier analysis
├── figures/
│   ├── fig_rq1_cv_stability.pdf              # + .png preview
│   ├── fig_rq2_permutation_importance.pdf
│   ├── fig_rq3_tempo_tier_popularity.pdf
│   ├── fig_rq4_genre_family.pdf
│   ├── fig_rq5_energy_danceability_interaction.pdf
│   ├── fig_rq6_imbalance_strategies.pdf
│   └── fig_rq7_acousticness_tiers.pdf
└── tables/
    ├── table_rq1_cv_stability.csv
    ├── table_rq2_permutation_importance.csv
    ├── table_rq3_tempo_tier_popularity.csv
    ├── table_rq4_genre_family.csv
    ├── table_rq5_energy_danceability_interaction.csv
    ├── table_rq6_imbalance_strategies.csv
    └── table_rq7_acousticness_tiers.csv
```

---

## Dataset

**Source file:** `tracks.csv` (Spotify Tracks Dataset)

The notebooks **auto-detect** whether they are running on Kaggle or locally:

- **On Kaggle:** the dataset is mounted at `/kaggle/input/.../tracks.csv` — no manual download needed. Just attach the dataset to your Kaggle notebook session.
- **Locally:** place `tracks.csv` in the same directory as the notebooks.

### Columns used

| Column | Type | Role |
|---|---|---|
| `popularity` | numeric (0–100) | source for binary target |
| `tempo` | numeric (BPM) | feature |
| `energy` | numeric (0–1) | feature |
| `danceability` | numeric (0–1) | feature |
| `valence` | numeric (0–1) | feature |
| `acousticness` | numeric (0–1) | feature |
| `liveness` | numeric (0–1) | feature |
| `instrumentalness` | numeric (0–1) | feature |
| `speechiness` | numeric (0–1) | feature |
| `key` | numeric (0–11) | feature |
| `mode` | binary (0/1) | feature |
| `time_signature` | numeric | feature |
| `genres` | list of strings | source for genre-family dummies |

---

## How to run

### Option A: Kaggle (recommended)

1. Go to https://www.kaggle.com and create a new notebook.
2. Attach the dataset (containing `tracks.csv`) to your notebook.
3. Upload one of the `RQ*.ipynb` files.
4. Click **Run All**. Outputs (figures and tables) are saved to `/kaggle/working/`.

### Option B: Local

1. Clone or download this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Place `tracks.csv` in the `notebooks/` folder.
4. Launch Jupyter: `jupyter notebook` and open any `RQ*.ipynb`.
5. Run all cells. Outputs are written to the current working directory.

---

## Research Questions

| RQ | Question |
|---|---|
| **RQ1** | How consistent are model performance estimates between a single stratified hold-out and 5-fold stratified cross-validation? |
| **RQ2** | Do gain-based and permutation-based feature importance methods agree on which audio features most drive popularity prediction? |
| **RQ3** | Does popularity rate and model predictability vary systematically across tempo tiers (Slow / Medium / Fast / Very Fast)? |
| **RQ4** | Does a model trained on the full dataset predict popularity equally well across different genre families (Pop, Rock, Hip-Hop, Electronic, Other)? |
| **RQ5** | How do energy and danceability jointly affect popularity rate and model predictability? |
| **RQ6** | How do class imbalance mitigation strategies affect the precision-recall trade-off given the strong skew toward non-popular tracks? |
| **RQ7** | Does acousticness tier (Electronic / Mixed / Acoustic) influence popularity predictability? |

---

## Dependencies

See `requirements.txt`. Key libraries: `pandas`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `matplotlib`, `scipy`.

---

## Author

Balla Shivaram — Matriculation Number: 89235214 — PS26 - DSC01 Machine Learning 120B

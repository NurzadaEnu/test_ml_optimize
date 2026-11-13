# How to run the pipeline, step by step

## Requirements

```bash
pip install -r requirements.txt
```

## Step 1 — Data generation (`generate_data.py`)

Synthetic dataset (3000 rows):
- **Volume:** `y1 = x1·η(x2,x3)`; η is least-squares calibrated on the three reference modes of Table 1; η ≤ 0.995, so `y1 < x1`.
- **Quality:** μ1–μ3 are reaction conversions (first-order plug-flow reactor, Arrhenius kinetics) calibrated at the centre of the operating ranges (x2 = 27.5, x3 = 321.5, x4 = 2.75, x5 = 350). The AorRA+PO solution is not used by the generator.

```bash
python3 generate_data.py
```

Expected output:
```
deterministic       : y1_table=76.0  y1_model=76.00
AorRA+PO (fuzzy)    : y1_table=77.5  y1_model=77.50
real_production     : y1_table=77.0  y1_model=77.00
```

## Step 2 — Training (`train_model.py`)

Four XGBoost models (n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9), 80:20 split. Prints test RMSE, R² and MAE → `data/test_metrics.csv`; plots `plots/parity_plots.png` and `plots/feature_importance.png`.

## Step 3 — Validation (`validate_model.py`)

- 5-fold CV: R² per fold, mean ± std → `data/cv_metrics.csv`. The synthetic samples are independent (no temporal autocorrelation), so random K-fold is appropriate.
- For real data, use `cross_validation(df, time_ordered=True)` (TimeSeriesSplit).
- Surrogate vs generator check on the Table 1 modes.

## Step 4 — Optimization (`optimize.py`)

Single problem (Eq. 16): `max y1` subject to `Score ≥ 0.90` and `y1 ≤ x1`, solved by random search over 60,000 points on the surrogate.
- verification of the recommended mode on the generator;
- `data/table1_common_model.csv`: all modes on one quality model;
- `plots/fig5_sensitivity_front.png`: (a) y1 and Score vs temperature, (b) ε-constraint front (`data/eps_front.csv`);
- leakage check: optimization with x2 fixed at 28 (the AorRA+PO value).

## Algorithm comparison (`compare_models.py`)

Eight algorithms, 5-fold CV → `data/model_comparison.csv`, `plots/model_comparison.png`.

## Full run

```bash
python3 generate_data.py && python3 train_model.py && python3 validate_model.py && python3 optimize.py
```

## Real plant data

Replace `data/synthetic_dataset.csv` (same columns: `x1_feed, x2_pressure, x3_temperature, x4_lhsv, x5_h2_ratio, y1_volume, mu1_unsaturates, mu2_sulfur, mu3_acids`), run steps 2–4, use a time-ordered split, and narrow `BOUNDS` to the historical ranges.

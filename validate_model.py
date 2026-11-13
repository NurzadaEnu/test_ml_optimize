"""
validate_model.py
==================
Validation of the XGBoost surrogates:
  1) Train/test 80:20 -- RMSE, R2, MAE                              (R2-11)
  2) 5-fold CV -- R2 mean +/- std across folds                      (R2-11)
     The synthetic samples are drawn independently and uniformly (no
     temporal autocorrelation), so random K-fold is appropriate. For real,
     time-ordered plant data a time-series (block) split is required.  (R2-7)
  3) Surrogate vs generator on the reference modes of Table 1.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

from generate_data import FEATURES, TARGETS, ANCHOR_POINTS, compute_y1, compute_mu_quality
from train_model import make_model


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def cross_validation(df, time_ordered=False):
    X = df[FEATURES].values
    cv = TimeSeriesSplit(n_splits=5) if time_ordered else KFold(5, shuffle=True, random_state=42)
    rows = []
    for target in TARGETS:
        s = cross_val_score(make_model(), X, df[target].values, cv=cv, scoring="r2")
        rows.append({"target": target, "R2_mean": s.mean(), "R2_std": s.std()})
        print(f"  {target:18s}  R2 per fold {np.round(s, 3)}  -> {s.mean():.4f} +/- {s.std():.4f}")
    return pd.DataFrame(rows)


def anchor_check():
    models = {t: joblib.load(f"models/xgb_{t}.joblib") for t in TARGETS}
    for _, r in ANCHOR_POINTS.iterrows():
        X = np.array([[r.x1, r.x2, r.x3, r.x4, r.x5]])
        y1s = models["y1_volume"].predict(X)[0]
        mus = [models[t].predict(X)[0] for t in TARGETS[1:]]
        mug = compute_mu_quality(r.x1, r.x2, r.x3, r.x4, r.x5)
        print(f"  {r.method:18s} y1: table={r.y1:.1f} gen={float(compute_y1(r.x1, r.x2, r.x3)):.2f} "
              f"surr={y1s:.2f} | mu gen={np.round(np.array(mug, float), 3)} surr={np.round(mus, 3)}")


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic_dataset.csv")
    section("1) Test metrics (80:20): see train_model.py / data/test_metrics.csv")
    print(pd.read_csv("data/test_metrics.csv").to_string(index=False))
    section("2) 5-fold CV (random; synthetic data have no autocorrelation)")
    cv = cross_validation(df)
    cv.to_csv("data/cv_metrics.csv", index=False)
    section("3) Reference modes of Table 1: generator vs surrogate")
    anchor_check()
    print("\nFor real (time-ordered) data use: cross_validation(df, time_ordered=True)")

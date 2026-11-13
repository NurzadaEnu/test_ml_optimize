"""
train_model.py
===============
Trains XGBoost surrogate models for y1, mu1, mu2, mu3.

Settings (stated in the paper, Section 3.3):
  * 80:20 train/test split (random_state=42)
  * XGBoost: n_estimators=600, max_depth=5, learning_rate=0.05,
             subsample=0.9, colsample_bytree=0.9, random_state=42
  * test-set metrics: RMSE, R2, MAE
  * 5-fold CV: R2 mean +/- std across folds (see validate_model.py)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from generate_data import FEATURES, TARGETS

XGB_PARAMS = dict(n_estimators=600, max_depth=5, learning_rate=0.05,
                  subsample=0.9, colsample_bytree=0.9,
                  random_state=42, n_jobs=-1, verbosity=0)

FEATURE_LABELS = {
    "x1_feed": "x1 feed", "x2_pressure": "x2 pressure", "x3_temperature": "x3 temperature",
    "x4_lhsv": "x4 LHSV", "x5_h2_ratio": "x5 H2/HC",
}


def make_model():
    return XGBRegressor(**XGB_PARAMS)


def train_all(df):
    X = df[FEATURES].values
    models, metrics = {}, {}
    for target in TARGETS:
        y = df[target].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = make_model().fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        rmse = mean_squared_error(y_te, y_pred) ** 0.5
        r2 = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        models[target] = model
        metrics[target] = {"rmse": rmse, "r2": r2, "mae": mae, "y_test": y_te, "y_pred": y_pred}
        print(f"{target:18s}  RMSE={rmse:.4f}   R2={r2:.4f}   MAE={mae:.4f}")
    return models, metrics


def plot_parity(metrics, path="plots/parity_plots.png"):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, target in zip(axes, TARGETS):
        m = metrics[target]
        ax.scatter(m["y_test"], m["y_pred"], alpha=0.35, s=14, color="#b45309")
        lims = [min(m["y_test"].min(), m["y_pred"].min()), max(m["y_test"].max(), m["y_pred"].max())]
        ax.plot(lims, lims, "--", color="#374151", linewidth=1)
        ax.set_title(f"{target}\nR2={m['r2']:.3f}  RMSE={m['rmse']:.3f}", fontsize=10)
        ax.set_xlabel("generator (test set)")
        ax.set_ylabel("XGBoost prediction")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_feature_importance(models, path="plots/feature_importance.png"):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, target in zip(axes, TARGETS):
        imp = models[target].feature_importances_
        labels = np.array([FEATURE_LABELS[f] for f in FEATURES])
        order = np.argsort(imp)
        ax.barh(labels[order], imp[order], color="#b45309")
        ax.set_title(target, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    df = pd.read_csv("data/synthetic_dataset.csv")
    print(f"Loaded {len(df)} rows\n")
    models, metrics = train_all(df)
    pd.DataFrame([{"target": t, "RMSE": m["rmse"], "R2": m["r2"], "MAE": m["mae"]}
                  for t, m in metrics.items()]).to_csv("data/test_metrics.csv", index=False)
    for target, model in models.items():
        joblib.dump(model, f"models/xgb_{target}.joblib")
    plot_parity(metrics)
    plot_feature_importance(models)
    print("\nModels saved to models/xgb_*.joblib, metrics -> data/test_metrics.csv")

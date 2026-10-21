"""
train_model.py
===============
Обучает ML-модели (Random Forest) для предсказания:
  - y1_obyem       -- объем гидрогенизата
  - mu1_nepredelnye, mu2_sera, mu3_kisloty -- степень выполнения требований качества

на синтетическом датасете (data/synthetic_dataset.csv).
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

FEATURES = ["x1_syre", "x2_davlenie", "x3_temperatura", "x4_skorost", "x5_vsg"]
TARGETS = ["y1_obyem", "mu1_nepredelnye", "mu2_sera", "mu3_kisloty"]
FEATURE_LABELS = {
    "x1_syre": "x1 сырьё", "x2_davlenie": "x2 давление", "x3_temperatura": "x3 температура",
    "x4_skorost": "x4 скорость подачи", "x5_vsg": "x5 ВСГ",
}


def train_all(df):
    X = df[FEATURES].values
    models, metrics = {}, {}

    for target in TARGETS:
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(
            n_estimators=400, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        # MAPE ненадежен, когда истинное значение близко к 0 (mu-показатели могут
        # быть ~0 при плохих режимах) -- считаем его только там, где y_test не мал.
        safe = np.abs(y_test) > 0.05
        mape = mean_absolute_percentage_error(y_test[safe], y_pred[safe]) * 100 if safe.sum() > 5 else np.nan

        models[target] = model
        metrics[target] = {"r2": r2, "mae": mae, "mape": mape,
                            "y_test": y_test, "y_pred": y_pred}
        mape_str = f"{mape:.2f}%" if not np.isnan(mape) else "н/д (значения близки к 0)"
        print(f"{target:20s}  R2={r2:.4f}   MAE={mae:.4f}   MAPE={mape_str}")

    return models, metrics


def plot_parity(metrics, path="plots/parity_plots.png"):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, target in zip(axes, TARGETS):
        m = metrics[target]
        ax.scatter(m["y_test"], m["y_pred"], alpha=0.35, s=14, color="#b45309")
        lims = [min(m["y_test"].min(), m["y_pred"].min()), max(m["y_test"].max(), m["y_pred"].max())]
        ax.plot(lims, lims, "--", color="#374151", linewidth=1)
        ax.set_title(f"{target}\nR2={m['r2']:.3f}  MAPE={m['mape']:.1f}%", fontsize=10)
        ax.set_xlabel("реальное (синтетика)")
        ax.set_ylabel("предсказание модели")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"Сохранено: {path}")


def plot_feature_importance(models, path="plots/feature_importance.png"):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, target in zip(axes, TARGETS):
        importances = models[target].feature_importances_
        labels = [FEATURE_LABELS[f] for f in FEATURES]
        order = np.argsort(importances)
        ax.barh(np.array(labels)[order], importances[order], color="#b45309")
        ax.set_title(target, fontsize=10)
        ax.set_xlim(0, max(importances) * 1.15)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"Сохранено: {path}")


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic_dataset.csv")
    print(f"Загружено {len(df)} строк из data/synthetic_dataset.csv\n")

    models, metrics = train_all(df)

    for target, model in models.items():
        joblib.dump(model, f"models/rf_{target}.joblib")

    plot_parity(metrics)
    plot_feature_importance(models)

    print("\nМодели сохранены в models/*.joblib")

"""
compare_models.py
==================
Сравнивает несколько ML-алгоритмов на датасете

Алгоритмы:
  - Linear Regression      -- простейший baseline (линейная зависимость)
  - Ridge Regression       -- линейная + регуляризация
  - KNN                    -- метод ближайших соседей
  - Random Forest          -- то, что уже использовалось в train_model.py
  - Gradient Boosting      -- sklearn, бустинг деревьев
  - XGBoost                -- популярный промышленный бустинг
  - LightGBM               -- быстрый бустинг от Microsoft
  - MLP (нейросеть)        -- простая полносвязная сеть

ЗАПУСК:
    python3 compare_models.py
Требует data/synthetic_dataset.csv
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, cross_validate

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

FEATURES = ["x1_syre", "x2_davlenie", "x3_temperatura", "x4_skorost", "x5_vsg"]
TARGETS = ["y1_obyem", "mu1_nepredelnye", "mu2_sera", "mu3_kisloty"]

MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "KNN (k=7)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)),
    "Random Forest": RandomForestRegressor(n_estimators=400, max_depth=12, min_samples_leaf=3,
                                            random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                    learning_rate=0.05, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                             random_state=42, n_jobs=-1, verbosity=0),
    "LightGBM": LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                               random_state=42, n_jobs=-1, verbosity=-1),
    "MLP (нейросеть)": make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42, early_stopping=True)
    ),
}


def compare_on_target(X, y, target_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in MODELS.items():
        t0 = time.time()
        scores = cross_validate(model, X, y, cv=kf,
                                 scoring=("r2", "neg_mean_absolute_error"))
        elapsed = time.time() - t0
        r2 = scores["test_r2"].mean()
        r2_std = scores["test_r2"].std()
        mae = -scores["test_neg_mean_absolute_error"].mean()
        rows.append({"target": target_name, "model": name, "R2_mean": r2,
                     "R2_std": r2_std, "MAE_mean": mae, "time_sec": elapsed})
    return rows


def main():
    df = pd.read_csv("data/synthetic_dataset.csv")
    X = df[FEATURES].values

    all_rows = []
    for target in TARGETS:
        print(f"\nСравнение моделей для: {target}")
        y = df[target].values
        rows = compare_on_target(X, y, target)
        rows_sorted = sorted(rows, key=lambda r: -r["R2_mean"])
        for r in rows_sorted:
            print(f"  {r['model']:20s}  R2={r['R2_mean']:.4f} (+/-{r['R2_std']:.4f})  "
                  f"MAE={r['MAE_mean']:.4f}  время={r['time_sec']:.2f}с")
        all_rows.extend(rows)

    results = pd.DataFrame(all_rows)
    results.to_csv("data/model_comparison.csv", index=False)
    print("\nПолная таблица сохранена -> data/model_comparison.csv")

    plot_comparison(results)


def plot_comparison(results):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    color_map = dict(zip(MODELS.keys(), colors))

    for ax, target in zip(axes, TARGETS):
        sub = results[results["target"] == target].sort_values("R2_mean")
        bar_colors = [color_map[m] for m in sub["model"]]
        ax.barh(sub["model"], sub["R2_mean"], xerr=sub["R2_std"], color=bar_colors)
        ax.set_title(target, fontsize=10)
        ax.set_xlabel("R2 (cross-val)")
        ax.set_xlim(min(0, sub["R2_mean"].min() - 0.05), 1.02)

    plt.tight_layout()
    plt.savefig("plots/model_comparison.png", dpi=130)
    print("Сохранено: plots/model_comparison.png")


if __name__ == "__main__":
    main()

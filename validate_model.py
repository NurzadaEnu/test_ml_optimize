"""
validate_model.py
==================
Валидация суррогатов XGBoost:
  1) Train/test 80:20 -- RMSE, R2, MAE                         (R2-11)
  2) 5-fold CV -- R2 mean +/- std по фолдам                    (R2-11)
     Синтетические выборки независимы и равномерны (нет автокорреляции),
     поэтому случайный K-fold корректен. Для реальных заводских данных
     нужен временной (блочный) сплит -- TimeSeriesSplit.            (R2-7)
  3) Сверка суррогата с генератором на опорных режимах таблицы 1.
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
        print(f"  {target:18s}  R2 по фолдам {np.round(s, 3)}  -> {s.mean():.4f} +/- {s.std():.4f}")
    return pd.DataFrame(rows)


def anchor_check():
    models = {t: joblib.load(f"models/xgb_{t}.joblib") for t in TARGETS}
    for _, r in ANCHOR_POINTS.iterrows():
        X = np.array([[r.x1, r.x2, r.x3, r.x4, r.x5]])
        y1s = models["y1_obyem"].predict(X)[0]
        mus = [models[t].predict(X)[0] for t in TARGETS[1:]]
        mug = compute_mu_quality(r.x1, r.x2, r.x3, r.x4, r.x5)
        print(f"  {r.method:18s} y1: table={r.y1:.1f} gen={float(compute_y1(r.x1, r.x2, r.x3)):.2f} "
              f"surr={y1s:.2f} | mu gen={np.round(np.array(mug, float), 3)} surr={np.round(mus, 3)}")


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic_dataset.csv")
    section("1) Test-метрики (80:20): см. train_model.py / data/test_metrics.csv")
    print(pd.read_csv("data/test_metrics.csv").to_string(index=False))
    section("2) 5-fold CV (случайный, синтетика без автокорреляции)")
    cv = cross_validation(df)
    cv.to_csv("data/cv_metrics.csv", index=False)
    section("3) Опорные режимы таблицы 1: генератор vs суррогат")
    anchor_check()
    print("\nДля реальных (упорядоченных во времени) данных: cross_validation(df, time_ordered=True)")

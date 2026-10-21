"""
optimize.py
===========
Использует  ML-модели (не исходные формулы генератора!) как
"черный ящик" и ищет режим (x1...x5), максимизирующий объем гидрогенизата
y1 при условии, что взвешенный критерий качества Score >= порога.

Score = beta1*mu1 + beta2*mu2 + beta3*mu3
Веса beta взяты из диссертации (важность требований для ЛПР): 0.35 / 0.40 / 0.25
"""

import joblib
import numpy as np
import pandas as pd

from generate_data import BOUNDS, ANCHOR_POINTS, MU_ANCHOR

FEATURES = ["x1_syre", "x2_davlenie", "x3_temperatura", "x4_skorost", "x5_vsg"]
BETA = {"mu1": 0.35, "mu2": 0.40, "mu3": 0.25}
SCORE_THRESHOLD = 0.90


def load_models():
    return {
        "y1": joblib.load("models/rf_y1_obyem.joblib"),
        "mu1": joblib.load("models/rf_mu1_nepredelnye.joblib"),
        "mu2": joblib.load("models/rf_mu2_sera.joblib"),
        "mu3": joblib.load("models/rf_mu3_kisloty.joblib"),
    }


def predict_all(models, X):
    """X: (n,5) массив в порядке x1..x5. Возвращает y1, mu1, mu2, mu3."""
    y1 = models["y1"].predict(X)
    mu1 = models["mu1"].predict(X)
    mu2 = models["mu2"].predict(X)
    mu3 = models["mu3"].predict(X)
    return y1, mu1, mu2, mu3


def random_search_optimize(models, n_samples=20000, threshold=SCORE_THRESHOLD, seed=7):
    """
    Простой, но честный и прозрачный метод: случайный перебор внутри
    разрешенных границ + фильтр по порогу качества + выбор максимума объема.
    (Тот самый механизм, который мы разбирали на примере таблицы вариантов A-D.)
    """
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.uniform(*BOUNDS["x1"], n_samples),
        rng.uniform(*BOUNDS["x2"], n_samples),
        rng.uniform(*BOUNDS["x3"], n_samples),
        rng.uniform(*BOUNDS["x4"], n_samples),
        rng.uniform(*BOUNDS["x5"], n_samples),
    ])

    y1, mu1, mu2, mu3 = predict_all(models, X)
    score = BETA["mu1"] * mu1 + BETA["mu2"] * mu2 + BETA["mu3"] * mu3

    feasible = score >= threshold
    if feasible.sum() == 0:
        raise RuntimeError(f"Ни один из {n_samples} вариантов не преодолел порог качества {threshold}")

    idx_feasible = np.where(feasible)[0]
    best_local = idx_feasible[np.argmax(y1[idx_feasible])]

    result = {
        "x1": X[best_local, 0], "x2": X[best_local, 1], "x3": X[best_local, 2],
        "x4": X[best_local, 3], "x5": X[best_local, 4],
        "y1": y1[best_local], "mu1": mu1[best_local], "mu2": mu2[best_local],
        "mu3": mu3[best_local], "score": score[best_local],
        "n_feasible": int(feasible.sum()), "n_total": n_samples,
    }
    return result


if __name__ == "__main__":
    models = load_models()
    result = random_search_optimize(models)

    print("=" * 70)
    print("ОПТИМИЗАЦИЯ (на основе ОБУЧЕННОЙ ML-модели, не исходных формул)")
    print("=" * 70)
    print(f"Из {result['n_total']} случайных вариантов допустимыми (Score>=0.90) "
          f"оказались {result['n_feasible']}\n")
    print(f"Найденный оптимальный режим:")
    print(f"  x1 (сырье)        = {result['x1']:.1f} м3/час")
    print(f"  x2 (давление)     = {result['x2']:.1f} кг/см2")
    print(f"  x3 (температура)  = {result['x3']:.1f} C")
    print(f"  x4 (скорость)     = {result['x4']:.2f} час^-1")
    print(f"  x5 (ВСГ)          = {result['x5']:.0f} нм3")
    print(f"\n  y1 (объем)  = {result['y1']:.2f} м3/час")
    print(f"  mu1={result['mu1']:.3f}  mu2={result['mu2']:.3f}  mu3={result['mu3']:.3f}")
    print(f"  Score = {result['score']:.3f}")

    print("\n" + "-" * 70)
    print("Сравнение с диссертацией (таблица 4.4):")
    print("-" * 70)
    comp = ANCHOR_POINTS.copy()
    comp.loc[len(comp)] = [result["x1"], result["x2"], result["x3"], result["x4"],
                            result["x5"], result["y1"], "ML-optimizer (this run)"]
    print(comp.to_string(index=False))

"""
optimize.py
===========
Единая задача оптимизации (Eq. 16 в статье) на суррогатах XGBoost:

    max  y1(x)
    s.t. Score(x) = 0.35*mu1 + 0.40*mu2 + 0.25*mu3 >= 0.90
         y1(x) <= x1                    (баланс объёма, явное ограничение)
         x in BOUNDS

Двухэтапная лексикографическая постановка (старое Eq. 17) удалена.
Дополнительно:
  * проверка рекомендованного режима на исходном генераторе            (R1-3)
  * сравнение режимов таблицы 1 на ОДНОЙ общей модели качества          (R1-5)
  * Рис. 5: чувствительность y1 и Score к температуре + фронт
    объём-качество (epsilon-constraint)                       (R1-2, R2-3, R2-4)
  * проверка утечки: давление фиксировано на значении AorRA+PO (28)     (R2-6)
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_data import BOUNDS, TARGETS, ANCHOR_POINTS, compute_y1, compute_mu_quality

BETA = np.array([0.35, 0.40, 0.25])
SCORE_THRESHOLD = 0.90
KEYS = ["x1", "x2", "x3", "x4", "x5"]


def load_models():
    return {t: joblib.load(f"models/xgb_{t}.joblib") for t in TARGETS}


def surrogate(models, X):
    y1 = models["y1_obyem"].predict(X)
    mu = np.column_stack([models[t].predict(X) for t in TARGETS[1:]])
    return y1, mu, mu @ BETA


def generator(X):
    X = np.atleast_2d(X)
    y1 = compute_y1(X[:, 0], X[:, 1], X[:, 2])
    mu = np.column_stack(compute_mu_quality(*[X[:, i] for i in range(5)]))
    return y1, mu, mu @ BETA


def sample(n, seed, fixed=None):
    rng = np.random.default_rng(seed)
    cols = []
    for k in KEYS:
        if fixed and k in fixed:
            cols.append(np.full(n, float(fixed[k])))
        else:
            cols.append(rng.uniform(*BOUNDS[k], n))
    return np.column_stack(cols)


def solve(models, eps=SCORE_THRESHOLD, n=60_000, seed=7, fixed=None, X=None):
    """max y1 s.t. Score>=eps, y1<=x1 (случайный поиск по суррогату)."""
    if X is None:
        X = sample(n, seed, fixed)
    y1, mu, score = surrogate(models, X)
    feas = (score >= eps) & (y1 <= X[:, 0])
    if not feas.any():
        return None
    i = np.where(feas)[0][np.argmax(y1[feas])]
    return {"x": X[i], "y1": y1[i], "mu": mu[i], "score": score[i], "n_feasible": int(feas.sum())}


def figure5(models, best, X_pool, path="plots/fig5_sensitivity_front.png"):
    T = np.linspace(*BOUNDS["x3"], 120)
    Xs = np.repeat(best["x"][None, :], len(T), 0)
    Xs[:, 2] = T
    y1g, _, sg = generator(Xs)
    y1s, _, ss = surrogate(models, Xs)

    eps_grid = np.round(np.arange(0.80, 0.971, 0.01), 2)
    front = []
    for e in eps_grid:
        r = solve(models, eps=e, X=X_pool)
        if r is not None:
            front.append((e, r["y1"]))
    front = np.array(front)

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3))
    a.plot(T, y1g, color="#1f4e79", label="y1, генератор")
    a.plot(T, y1s, "--", color="#1f4e79", alpha=.6, label="y1, XGBoost")
    a.set_xlabel("Температура x3, °C"); a.set_ylabel("y1, м³/ч")
    a2 = a.twinx()
    a2.plot(T, sg, color="#b45309", label="Score, генератор")
    a2.plot(T, ss, "--", color="#b45309", alpha=.6, label="Score, XGBoost")
    a2.axhline(SCORE_THRESHOLD, color="gray", lw=.8, ls=":")
    a2.set_ylabel("Score")
    a.axvline(best["x"][2], color="k", lw=.8, ls=":")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    a.set_title("(a) Чувствительность к температуре")
    b.plot(front[:, 0], front[:, 1], "o-", color="#1f4e79")
    b.set_xlabel("Порог качества ε (Score ≥ ε)"); b.set_ylabel("max y1, м³/ч")
    b.set_title("(b) Фронт объём–качество (ε-constraint)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)
    pd.DataFrame(front, columns=["eps", "y1_max"]).to_csv("data/eps_front.csv", index=False)
    print(f"Сохранено: {path}, data/eps_front.csv")


def fmt(x):
    return f"x1={x[0]:.1f} x2={x[1]:.1f} x3={x[2]:.1f} x4={x[3]:.2f} x5={x[4]:.0f}"


if __name__ == "__main__":
    models = load_models()
    X_pool = sample(60_000, 7)
    best = solve(models, X=X_pool)

    print("=" * 74 + "\nРЕКОМЕНДОВАННЫЙ РЕЖИМ (суррогат XGBoost, Eq. 16)\n" + "=" * 74)
    print(f"Допустимых вариантов: {best['n_feasible']} из {len(X_pool)}")
    print(f"  {fmt(best['x'])}")
    print(f"  суррогат : y1={best['y1']:.2f}  mu={np.round(best['mu'], 3)}  Score={best['score']:.3f}")
    y1g, mug, sg = generator(best["x"])
    print(f"  генератор: y1={y1g[0]:.2f}  mu={np.round(mug[0], 3)}  Score={sg[0]:.3f}")
    print(f"  y1 <= x1 : {y1g[0]:.2f} <= {best['x'][0]:.1f}  -> {bool(y1g[0] <= best['x'][0])}")
    print(f"  расхождение суррогат-генератор по y1: {abs(best['y1'] - y1g[0]):.2f} м3/ч")

    print("\n" + "-" * 74 + "\nТаблица 1: все режимы на ОДНОЙ общей модели качества (генератор)\n" + "-" * 74)
    rows = []
    for _, r in ANCHOR_POINTS.iterrows():
        x = np.array([r.x1, r.x2, r.x3, r.x4, r.x5], float)
        y, m, s = generator(x)
        rows.append([r.method, *x, y[0], *m[0], s[0]])
    rows.append(["XGBoost + Eq.16", *best["x"], y1g[0], *mug[0], sg[0]])
    tab = pd.DataFrame(rows, columns=["method", *KEYS, "y1", "mu1", "mu2", "mu3", "Score"])
    print(tab.round(3).to_string(index=False))
    tab.to_csv("data/table1_common_model.csv", index=False)

    print("\n" + "-" * 74 + "\nПроверка утечки (R2-6): x2 фиксировано = 28 (как в AorRA+PO)\n" + "-" * 74)
    r28 = solve(models, fixed={"x2": 28.0}, seed=11)
    if r28 is None:
        print("  при x2=28 нет режимов со Score >= 0.90")
    else:
        print(f"  {fmt(r28['x'])}  y1={r28['y1']:.2f}  Score={r28['score']:.3f}")

    figure5(models, best, X_pool)

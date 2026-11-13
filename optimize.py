"""
optimize.py
===========
Single optimization problem (Eq. 16 in the paper) on the XGBoost surrogates:

    max  y1(x)
    s.t. Score(x) = 0.35*mu1 + 0.40*mu2 + 0.25*mu3 >= 0.90
         y1(x) <= x1                    (volume balance, explicit constraint)
         x in BOUNDS

The two-stage lexicographic formulation (former Eq. 17) has been removed.
In addition:
  * verification of the recommended mode on the original generator     (R1-3)
  * all Table 1 modes evaluated on ONE common quality model            (R1-5)
  * Fig. 5: sensitivity of y1 and Score to temperature + volume-quality
    front (epsilon-constraint)                                (R1-2, R2-3, R2-4)
  * leakage check: pressure fixed at the AorRA+PO value (28)           (R2-6)
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
    y1 = models[TARGETS[0]].predict(X)
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
    """max y1 s.t. Score>=eps, y1<=x1 (random search on the surrogate)."""
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
    a.plot(T, y1g, color="#1f4e79", label="y1, generator")
    a.plot(T, y1s, "--", color="#1f4e79", alpha=.6, label="y1, XGBoost")
    a.set_xlabel("Temperature x3, °C"); a.set_ylabel("y1, m³/h")
    a2 = a.twinx()
    a2.plot(T, sg, color="#b45309", label="Score, generator")
    a2.plot(T, ss, "--", color="#b45309", alpha=.6, label="Score, XGBoost")
    a2.axhline(SCORE_THRESHOLD, color="gray", lw=.8, ls=":")
    a2.set_ylabel("Score")
    a.axvline(best["x"][2], color="k", lw=.8, ls=":")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    a.set_title("(a) Sensitivity to temperature")
    b.plot(front[:, 0], front[:, 1], "o-", color="#1f4e79")
    b.set_xlabel("Quality threshold ε (Score ≥ ε)"); b.set_ylabel("max y1, m³/h")
    b.set_title("(b) Volume–quality front (ε-constraint)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)
    pd.DataFrame(front, columns=["eps", "y1_max"]).to_csv("data/eps_front.csv", index=False)
    print(f"Saved: {path}, data/eps_front.csv")


def fmt(x):
    return f"x1={x[0]:.1f} x2={x[1]:.1f} x3={x[2]:.1f} x4={x[3]:.2f} x5={x[4]:.0f}"


def banner(title, ch="="):
    print("\n" + ch * 74 + "\n" + title + "\n" + ch * 74)


if __name__ == "__main__":
    models = load_models()
    X_pool = sample(60_000, 7)
    best = solve(models, X=X_pool)

    banner("RECOMMENDED MODE (XGBoost surrogate, Eq. 16)")
    print(f"Feasible candidates: {best['n_feasible']} of {len(X_pool)}")
    print(f"  {fmt(best['x'])}")
    print(f"  surrogate: y1={best['y1']:.2f}  mu={np.round(best['mu'], 3)}  Score={best['score']:.3f}")
    y1g, mug, sg = generator(best["x"])
    print(f"  generator: y1={y1g[0]:.2f}  mu={np.round(mug[0], 3)}  Score={sg[0]:.3f}")
    print(f"  y1 <= x1 : {y1g[0]:.2f} <= {best['x'][0]:.1f}  -> {bool(y1g[0] <= best['x'][0])}")
    print(f"  surrogate-generator discrepancy in y1: {abs(best['y1'] - y1g[0]):.2f} m3/h")

    banner("Table 1: all modes on ONE common quality model (generator)", "-")
    rows = []
    for _, r in ANCHOR_POINTS.iterrows():
        x = np.array([r.x1, r.x2, r.x3, r.x4, r.x5], float)
        y, m, s = generator(x)
        rows.append([r.method, *x, y[0], *m[0], s[0]])
    rows.append(["XGBoost + Eq.16", *best["x"], y1g[0], *mug[0], sg[0]])
    tab = pd.DataFrame(rows, columns=["method", *KEYS, "y1", "mu1", "mu2", "mu3", "Score"])
    print(tab.round(3).to_string(index=False))
    tab.to_csv("data/table1_common_model.csv", index=False)

    banner("Leakage check (R2-6): x2 fixed = 28 (as in AorRA+PO)", "-")
    r28 = solve(models, fixed={"x2": 28.0}, seed=11)
    if r28 is None:
        print("  no mode with Score >= 0.90 at x2 = 28")
    else:
        print(f"  {fmt(r28['x'])}  y1={r28['y1']:.2f}  Score={r28['score']:.3f}")

    figure5(models, best, X_pool)

"""
generate_data.py
=================
Генератор СИНТЕТИЧЕСКОГО датасета для реактора гидроочистки Р-1
(установка ЛГ-35-11/300-95) — версия после рецензирования.

Что изменено по замечаниям рецензентов:
  * R1-1  Объём гидрогенизата задаётся балансом выхода:  y1 = x1 * eta(x2, x3),
          eta откалибрована МНК по трём опорным режимам таблицы 1
          (76.0 / 77.5 / 77.0 м3/ч). eta <= 0.995, поэтому y1 < x1 всегда.
  * R2-6  Функции принадлежности качества НЕ центрированы на решении AorRA+PO.
          mu1..mu3 = степени превращения из кинетики Аррениуса, откалиброванной
          в ЦЕНТРЕ рабочих диапазонов. Генератор не содержит информации об оптимуме.
  * R1-3  mu = степень превращения: больше mu -> глубже гидрирование /
          обессеривание / удаление кислых соединений.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Границы режимных параметров
# ---------------------------------------------------------------------------
BOUNDS = {
    "x1": (45, 80),    # сырье, м3/ч
    "x2": (20, 35),    # давление, кг/см2
    "x3": (300, 343),  # температура, C
    "x4": (0.5, 5),    # объемная скорость подачи, ч^-1
    "x5": (200, 500),  # ВСГ, H2/УВ, нм3
}
FEATURES = ["x1_syre", "x2_davlenie", "x3_temperatura", "x4_skorost", "x5_vsg"]
TARGETS = ["y1_obyem", "mu1_nepredelnye", "mu2_sera", "mu3_kisloty"]

# Центр рабочих диапазонов -- точка калибровки кинетики (НЕ оптимум!)
CENTER = {k: 0.5 * (lo + hi) for k, (lo, hi) in BOUNDS.items()}

R_GAS = 8.314  # Дж/(моль*К)

# Типовые литературные кинетические параметры (представительные, не заводские).
#   Ea  -- энергия активации, Дж/моль
#   p,q -- порядки по давлению и по H2/УВ
#   conv_c -- степень превращения в центре диапазонов (нейтральная калибровка)
KINETICS = {
    "mu1": {"Ea": 50_000.0, "p": 0.3, "q": 0.7, "conv_c": 0.50},  # гидрирование непредельных
    "mu2": {"Ea": 80_000.0, "p": 0.5, "q": 0.5, "conv_c": 0.50},  # HDS (сера)
    "mu3": {"Ea": 65_000.0, "p": 0.4, "q": 0.6, "conv_c": 0.50},  # удаление кислых соединений
}

# ---------------------------------------------------------------------------
# 2. Опорные режимы таблицы 1 (используются ТОЛЬКО для баланса выхода y1)
# ---------------------------------------------------------------------------
ANCHOR_POINTS = pd.DataFrame([
    # x1,  x2,  x3,  x4,  x5,   y1,   метод
    [82, 31, 345, 3, 400, 76.0, "deterministic"],
    [80, 28, 337, 3, 400, 77.5, "AorRA+PO (fuzzy)"],
    [80, 30, 340, 3, 400, 77.0, "real_production"],
], columns=["x1", "x2", "x3", "x4", "x5", "y1", "method"])

ETA_MIN, ETA_MAX = 0.85, 0.995


def _fit_yield_loss_coeffs():
    """
    МНК-калибровка доли потерь (крекинг при избыточной жесткости) по 3 опорным режимам:
        eta = 1 - L0 - L1*(x3-330) - L3*(x2-28),   y1 = x1 * eta
    """
    A, b = [], []
    for _, r in ANCHOR_POINTS.iterrows():
        A.append([1.0, r["x3"] - 330.0, r["x2"] - 28.0])
        b.append(1.0 - r["y1"] / r["x1"])
    coeffs, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return tuple(coeffs)


L0, L1, L3 = _fit_yield_loss_coeffs()


def yield_fraction(x2, x3):
    """Доля сырья, переходящая в жидкий гидрогенизат (eta < 1 => y1 < x1)."""
    frac = 1 - L0 - L1 * (np.asarray(x3) - 330.0) - L3 * (np.asarray(x2) - 28.0)
    return np.clip(frac, ETA_MIN, ETA_MAX)


def compute_y1(x1, x2, x3, x4=None, x5=None):
    return np.asarray(x1) * yield_fraction(x2, x3)


def _rate_constant(x2, x3, x5, Ea, p, q):
    """Аррениус, приведённый к центру диапазонов: k/k_c."""
    T = np.asarray(x3) + 273.15
    Tc = CENTER["x3"] + 273.15
    return (np.exp(-Ea / R_GAS * (1.0 / T - 1.0 / Tc))
            * (np.asarray(x2) / CENTER["x2"]) ** p
            * (np.asarray(x5) / CENTER["x5"]) ** q)


def _conversion(x2, x3, x4, x5, Ea, p, q, conv_c):
    """Реактор идеального вытеснения 1-го порядка: X = 1 - exp(-k*tau), tau ~ 1/LHSV.
    Множитель k_c*tau_c выбирается так, чтобы в центре диапазонов X = conv_c."""
    kc_tauc = -np.log(1.0 - conv_c)
    tau_rel = CENTER["x4"] / np.asarray(x4)
    k_rel = _rate_constant(x2, x3, x5, Ea, p, q)
    return np.clip(1.0 - np.exp(-kc_tauc * k_rel * tau_rel), 0.0, 1.0)


def compute_mu_quality(x1, x2, x3, x4, x5):
    out = []
    for key in ("mu1", "mu2", "mu3"):
        kp = KINETICS[key]
        out.append(_conversion(x2, x3, x4, x5, kp["Ea"], kp["p"], kp["q"], kp["conv_c"]))
    return tuple(out)


def generate_dataset(n_samples=3000, noise_y1_pct=0.03, noise_mu_abs=0.02, seed=42):
    rng = np.random.default_rng(seed)
    x = {k: rng.uniform(*BOUNDS[k], n_samples) for k in BOUNDS}

    y1 = compute_y1(x["x1"], x["x2"], x["x3"])
    mu1, mu2, mu3 = compute_mu_quality(x["x1"], x["x2"], x["x3"], x["x4"], x["x5"])

    # Шум (аналог погрешности приборов / лабораторных анализов)
    y1 = y1 * (1 + rng.normal(0, noise_y1_pct, n_samples))
    mu1 = np.clip(mu1 + rng.normal(0, noise_mu_abs, n_samples), 0, 1)
    mu2 = np.clip(mu2 + rng.normal(0, noise_mu_abs, n_samples), 0, 1)
    mu3 = np.clip(mu3 + rng.normal(0, noise_mu_abs, n_samples), 0, 1)

    return pd.DataFrame({
        "x1_syre": x["x1"], "x2_davlenie": x["x2"], "x3_temperatura": x["x3"],
        "x4_skorost": x["x4"], "x5_vsg": x["x5"],
        "y1_obyem": y1, "mu1_nepredelnye": mu1, "mu2_sera": mu2, "mu3_kisloty": mu3,
    })


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(n_samples=3000)
    df.to_csv("data/synthetic_dataset.csv", index=False)
    print(f"Сгенерировано {len(df)} строк -> data/synthetic_dataset.csv")
    print(f"\nБаланс выхода: L0={L0:.6f}  L1={L1:.6f}  L3={L3:.6f}")
    print("Проверка на опорных режимах таблицы 1:")
    for _, r in ANCHOR_POINTS.iterrows():
        print(f"  {r.method:20s}: y1_table={r.y1:.1f}  y1_model={compute_y1(r.x1, r.x2, r.x3):.2f}")
    print(f"\nКалибровка кинетики в центре диапазонов: {CENTER}")
    print(f"max(y1/x1) в датасете без шума: {yield_fraction(df.x2_davlenie, df.x3_temperatura).max():.3f} (< 1)")

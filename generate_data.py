"""
generate_data.py
=================
Генератор СИНТЕТИЧЕСКОГО датасета для реактора гидроочистки Р-1
(установка ЛГ-35-11/300-95).

"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Границы режимных параметров (из диссертации / скриншота карты опроса)
# ---------------------------------------------------------------------------
BOUNDS = {
    "x1": (45, 80),    # сырье, м3/час
    "x2": (20, 35),    # давление, кг/см2
    "x3": (300, 343),  # температура, C
    "x4": (0.5, 5),    # объемная скорость подачи, час^-1
    "x5": (200, 500),  # ВСГ, H2/УВ, нм3
}

R_GAS = 8.314  # Дж/(моль*К)

# Типовые (представительные, НЕ заводские) кинетические константы HDS/HYD
KINETICS = {
    "Ea_S": 80_000.0,   # Дж/моль, энергия активации HDS (лит. диапазон ~60-100 кДж/моль)
    "p_S": 0.5,         # порядок по давлению для HDS
    "q_S": 0.5,         # порядок по H2/УВ для HDS
    "Ea_H": 50_000.0,   # Дж/моль, энергия активации гидрирования непредельных (обычно ниже, чем HDS)
    "p_H": 0.3,
    "q_H": 0.7,
}

# ---------------------------------------------------------------------------
# 2. Реальные опорные точки из диссертации (таблица сравнения методов, раздел 4)
#    Используются для калибровки модели потерь объема (крекинг).
# ---------------------------------------------------------------------------
ANCHOR_POINTS = pd.DataFrame([
    # x1,  x2,  x3,  x4,  x5,   y1 (факт/расчет), метод
    [82, 31, 345, 3, 400, 76.5, "deterministic"],
    [80, 28, 337, 3, 400, 78.0, "AorRA+PO (proposed)"],
    [80, 30, 340, 3, 400, 77.5, "real_production"],
], columns=["x1", "x2", "x3", "x4", "x5", "y1", "method"])

# Точка с известными значениями функций принадлежности качества (только одна!)
MU_ANCHOR = {"x1": 80, "x2": 28, "x3": 337, "x4": 3, "x5": 400,
             "mu1": 1.00, "mu2": 1.00, "mu3": 0.98}


def _fit_yield_loss_coeffs():
    """
    Калибрует линейную модель потерь объема (крекинг при избыточной жесткости)
    методом наименьших квадратов по 3 реальным точкам таблицы 4.4.

    yield_frac = 1 - L0 - L1*(x3-330) - L3*(x2-28)
    y1_pred    = x1 * yield_frac
    """
    A = []
    b = []
    for _, row in ANCHOR_POINTS.iterrows():
        yield_frac = row["y1"] / row["x1"]
        A.append([1.0, (row["x3"] - 330.0), (row["x2"] - 28.0)])
        b.append(1.0 - yield_frac)
    A = np.array(A)
    b = np.array(b)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    L0, L1, L3 = coeffs
    return L0, L1, L3


L0, L1, L3 = _fit_yield_loss_coeffs()


def yield_fraction(x2, x3):
    """Доля сырья, сохраняющаяся в жидком продукте (остальное -- потери в газ при крекинге)."""
    frac = 1 - L0 - L1 * (x3 - 330.0) - L3 * (x2 - 28.0)
    # Ограничиваем физически разумным диапазоном -- за пределами калибровочных
    # точек линейная модель ненадежна (риск экстраполяции, о котором мы говорили).
    return np.clip(frac, 0.85, 0.995)


def compute_y1(x1, x2, x3, x4, x5):
    return x1 * yield_fraction(x2, x3)


def _arrhenius_conversion(x2, x3, x4, x5, Ea, p, q):
    T_K = x3 + 273.15
    k = np.exp(-Ea / (R_GAS * T_K)) * (x2 / 28.0) ** p * (x5 / 400.0) ** q
    tau = 1.0 / x4  # время контакта, пропорционально 1/LHSV
    conv = 1 - np.exp(-k * 1e7 * tau)  # 1e7 -- масштабирующий множитель (A_0), подобран
    return np.clip(conv, 0, 1)


def compute_mu_quality(x1, x2, x3, x4, x5):
    """
    mu1 -- непредельные углеводороды (гидрирование, чувствительно к давлению/H2)
    mu2 -- сера (HDS, самое важное требование, chувствительно к температуре/времени контакта/H2)
    mu3 -- кислоты/щелочи (промывка, чувствительно к температуре/H2)
    Гауссовы функции принадлежности, откалиброванные так, чтобы точно давать
    известные значения в MU_ANCHOR.
    """
    mu1 = np.exp(-((x2 - 28) / 9) ** 2) * np.exp(-((x5 - 400) / 180) ** 2)
    mu2 = (np.exp(-((x3 - 337) / 24) ** 2)
           * np.exp(-((x4 - 3) / 2.2) ** 2)
           * np.exp(-((x5 - 400) / 200) ** 2))
    mu3 = 0.98 * np.exp(-((x3 - 337) / 34) ** 2) * np.exp(-((x5 - 400) / 150) ** 2)
    return mu1, mu2, mu3


def generate_dataset(n_samples=3000, noise_y1_pct=0.03, noise_mu_abs=0.02, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(*BOUNDS["x1"], n_samples)
    x2 = rng.uniform(*BOUNDS["x2"], n_samples)
    x3 = rng.uniform(*BOUNDS["x3"], n_samples)
    x4 = rng.uniform(*BOUNDS["x4"], n_samples)
    x5 = rng.uniform(*BOUNDS["x5"], n_samples)

    y1_clean = compute_y1(x1, x2, x3, x4, x5)
    mu1, mu2, mu3 = compute_mu_quality(x1, x2, x3, x4, x5)

    # Реалистичный шум (аналог погрешности лабораторных анализов / приборов)
    y1 = y1_clean * (1 + rng.normal(0, noise_y1_pct, n_samples))
    mu1 = np.clip(mu1 + rng.normal(0, noise_mu_abs, n_samples), 0, 1)
    mu2 = np.clip(mu2 + rng.normal(0, noise_mu_abs, n_samples), 0, 1)
    mu3 = np.clip(mu3 + rng.normal(0, noise_mu_abs, n_samples), 0, 1)

    df = pd.DataFrame({
        "x1_syre": x1, "x2_davlenie": x2, "x3_temperatura": x3,
        "x4_skorost": x4, "x5_vsg": x5,
        "y1_obyem": y1, "mu1_nepredelnye": mu1, "mu2_sera": mu2, "mu3_kisloty": mu3,
    })
    return df


if __name__ == "__main__":
    df = generate_dataset(n_samples=3000)
    df.to_csv("data/synthetic_dataset.csv", index=False)
    print(f"Сгенерировано {len(df)} строк -> data/synthetic_dataset.csv")
    print("\nКалибровочные коэффициенты модели потерь объема:")
    print(f"  L0={L0:.6f}  L1={L1:.6f}  L3={L3:.6f}")
    print("\nПроверка на опорных точках (должны совпадать с диссертацией):")
    for _, row in ANCHOR_POINTS.iterrows():
        pred = compute_y1(row.x1, row.x2, row.x3, row.x4, row.x5)
        print(f"  {row.method:22s}: y1_real={row.y1:.1f}  y1_model={pred:.2f}")
    mu1a, mu2a, mu3a = compute_mu_quality(**{k: MU_ANCHOR[k] for k in ["x1", "x2", "x3", "x4", "x5"]})
    print(f"\n  mu-anchor: real=(1.00, 1.00, 0.98)  model=({mu1a:.3f}, {mu2a:.3f}, {mu3a:.3f})")

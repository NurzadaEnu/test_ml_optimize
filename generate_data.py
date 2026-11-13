"""
generate_data.py
=================
Generator of the SYNTHETIC dataset for hydrotreating reactor R-1
(unit LG-35-11/300-95) -- revised version after peer review.

Changes made in response to the reviewers:
  * R1-1  Hydrogenate volume follows a yield balance:  y1 = x1 * eta(x2, x3),
          eta is least-squares calibrated on the three reference modes of
          Table 1 (76.0 / 77.5 / 77.0 m3/h). eta <= 0.995, hence y1 < x1 always.
  * R2-6  Quality membership functions are NOT centred on the AorRA+PO solution.
          mu1..mu3 are reaction conversions from Arrhenius kinetics calibrated
          at the CENTRE of the operating ranges. The generator contains no
          information about the optimum.
  * R1-3  mu = conversion: a higher mu means deeper hydrogenation /
          desulfurization / removal of acidic compounds.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Operating-parameter bounds
# ---------------------------------------------------------------------------
BOUNDS = {
    "x1": (45, 80),    # feed rate, m3/h
    "x2": (20, 35),    # pressure, kgf/cm2
    "x3": (300, 343),  # temperature, C
    "x4": (0.5, 5),    # liquid hourly space velocity (LHSV), h^-1
    "x5": (200, 500),  # hydrogen-containing gas, H2/HC ratio, nm3
}
FEATURES = ["x1_feed", "x2_pressure", "x3_temperature", "x4_lhsv", "x5_h2_ratio"]
TARGETS = ["y1_volume", "mu1_unsaturates", "mu2_sulfur", "mu3_acids"]

# Centre of the operating ranges -- kinetics calibration point (NOT the optimum!)
CENTER = {k: 0.5 * (lo + hi) for k, (lo, hi) in BOUNDS.items()}

R_GAS = 8.314  # J/(mol*K)

# Typical literature kinetic parameters (representative, not plant-specific).
#   Ea     -- activation energy, J/mol
#   p, q   -- reaction orders in pressure and in H2/HC ratio
#   conv_c -- conversion at the centre of the ranges (neutral calibration)
KINETICS = {
    "mu1": {"Ea": 50_000.0, "p": 0.3, "q": 0.7, "conv_c": 0.50},  # hydrogenation of unsaturates
    "mu2": {"Ea": 80_000.0, "p": 0.5, "q": 0.5, "conv_c": 0.50},  # hydrodesulfurization (HDS)
    "mu3": {"Ea": 65_000.0, "p": 0.4, "q": 0.6, "conv_c": 0.50},  # removal of acidic compounds
}

# ---------------------------------------------------------------------------
# 2. Reference modes of Table 1 (used ONLY for the y1 yield balance)
# ---------------------------------------------------------------------------
ANCHOR_POINTS = pd.DataFrame([
    # x1,  x2,  x3,  x4,  x5,   y1,   method
    [82, 31, 345, 3, 400, 76.0, "deterministic"],
    [80, 28, 337, 3, 400, 77.5, "AorRA+PO (fuzzy)"],
    [80, 30, 340, 3, 400, 77.0, "real_production"],
], columns=["x1", "x2", "x3", "x4", "x5", "y1", "method"])

ETA_MIN, ETA_MAX = 0.85, 0.995


def _fit_yield_loss_coeffs():
    """
    Least-squares calibration of the loss fraction (cracking at excessive
    severity) on the 3 reference modes:
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
    """Fraction of feed recovered as liquid hydrogenate (eta < 1 => y1 < x1)."""
    frac = 1 - L0 - L1 * (np.asarray(x3) - 330.0) - L3 * (np.asarray(x2) - 28.0)
    return np.clip(frac, ETA_MIN, ETA_MAX)


def compute_y1(x1, x2, x3, x4=None, x5=None):
    return np.asarray(x1) * yield_fraction(x2, x3)


def _rate_constant(x2, x3, x5, Ea, p, q):
    """Arrhenius rate constant relative to the centre of the ranges: k/k_c."""
    T = np.asarray(x3) + 273.15
    Tc = CENTER["x3"] + 273.15
    return (np.exp(-Ea / R_GAS * (1.0 / T - 1.0 / Tc))
            * (np.asarray(x2) / CENTER["x2"]) ** p
            * (np.asarray(x5) / CENTER["x5"]) ** q)


def _conversion(x2, x3, x4, x5, Ea, p, q, conv_c):
    """First-order plug-flow reactor: X = 1 - exp(-k*tau), tau ~ 1/LHSV.
    k_c*tau_c is chosen so that X = conv_c at the centre of the ranges."""
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

    # Noise (analogue of instrument / laboratory analysis error)
    y1 = y1 * (1 + rng.normal(0, noise_y1_pct, n_samples))
    mu1 = np.clip(mu1 + rng.normal(0, noise_mu_abs, n_samples), 0, 1)
    mu2 = np.clip(mu2 + rng.normal(0, noise_mu_abs, n_samples), 0, 1)
    mu3 = np.clip(mu3 + rng.normal(0, noise_mu_abs, n_samples), 0, 1)

    return pd.DataFrame({
        "x1_feed": x["x1"], "x2_pressure": x["x2"], "x3_temperature": x["x3"],
        "x4_lhsv": x["x4"], "x5_h2_ratio": x["x5"],
        "y1_volume": y1, "mu1_unsaturates": mu1, "mu2_sulfur": mu2, "mu3_acids": mu3,
    })


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(n_samples=3000)
    df.to_csv("data/synthetic_dataset.csv", index=False)
    print(f"Generated {len(df)} rows -> data/synthetic_dataset.csv")
    print(f"\nYield balance: L0={L0:.6f}  L1={L1:.6f}  L3={L3:.6f}")
    print("Check on the reference modes of Table 1:")
    for _, r in ANCHOR_POINTS.iterrows():
        print(f"  {r.method:20s}: y1_table={r.y1:.1f}  y1_model={compute_y1(r.x1, r.x2, r.x3):.2f}")
    print(f"\nKinetics calibrated at the centre of the ranges: {CENTER}")
    print(f"max(y1/x1) in the noise-free dataset: "
          f"{yield_fraction(df.x2_pressure, df.x3_temperature).max():.3f} (< 1)")

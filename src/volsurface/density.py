"""Risk-neutral density (RND) from the fitted SVI surface.

Breeden–Litzenberger: the risk-neutral density of the underlying is the (discounted)
second derivative of the call price in strike, ``f(K) = e^{rT} ∂²C/∂K²``. Taking that
derivative on raw market quotes is hopelessly noisy; instead we use the **closed-form**
density implied by a fitted SVI slice (Gatheral, *The Volatility Surface*):

    p(k) = g(k) / √(2π·w(k)) · exp(-½ · d₋(k)²),   d₋(k) = -k/√w - √w/2,

where ``g(k)`` is the same function that certifies the slice is butterfly-free, ``w(k)``
is SVI total variance, and ``k = log(K/F)``. ``p(k)`` is the density of the log-return
``log(S_T/F)`` (integrates to 1 in k); the strike-space density is ``f(K) = p(k)/K``.

Because ``p`` shares ``g`` with the butterfly check, a butterfly-free slice has ``p ≥ 0``
everywhere and ``∫p dk = 1`` — a self-consistent, non-negative implied distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402
from .svi import SVIParams, g_function, params_from_row  # noqa: E402

_SQRT_2PI = np.sqrt(2.0 * np.pi)


@dataclass
class DensityStats:
    total_prob: float   # ∫ p(k) dk over the evaluation grid (should be ≈ 1)
    min_density: float  # min p(k) (should be ≥ 0 for an arbitrage-free slice)


def rnd_logmoneyness(params: SVIParams, k: np.ndarray) -> np.ndarray:
    """Risk-neutral density of the log-return log(S_T/F), evaluated at log-moneyness k."""
    k = np.asarray(k, dtype="float64")
    w = np.maximum(np.asarray(params.total_variance(k), dtype="float64"), 1e-12)
    sqrt_w = np.sqrt(w)
    d_minus = -k / sqrt_w - 0.5 * sqrt_w
    g = g_function(params, k)
    return g / (sqrt_w * _SQRT_2PI) * np.exp(-0.5 * d_minus * d_minus)


def rnd_strike(params: SVIParams, F: float, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strike-space density: returns (K, f(K)) with K = F·e^k and f(K) = p(k)/K."""
    k = np.asarray(k, dtype="float64")
    K = F * np.exp(k)
    f_K = rnd_logmoneyness(params, k) / K
    return K, f_K


def density_stats(params: SVIParams, k_grid: np.ndarray) -> DensityStats:
    """Integrate the density over k (trapezoid) and report its minimum."""
    p = rnd_logmoneyness(params, k_grid)
    total = float(np.trapz(p, k_grid))
    return DensityStats(total_prob=total, min_density=float(np.min(p)))


def plot_rnd(
    svi_df: pd.DataFrame,
    expiries=None,
    cfg: Config | None = None,
    name: str = "risk_neutral_density.png",
    k_range: tuple[float, float] = (-0.6, 0.4),
) -> Path:
    """Plot strike-space RND for a few expiries ('what the market implies for S_T')."""
    cfg = cfg or Config.load()
    svi_df = svi_df.sort_values("T")
    if expiries is None:
        # near / mid / far representative expiries
        idx = sorted({0, len(svi_df) // 2, len(svi_df) - 1})
        rows = [svi_df.iloc[i] for i in idx]
    else:
        rows = [svi_df[svi_df["expiry"] == e].iloc[0] for e in expiries]

    k = np.linspace(k_range[0], k_range[1], 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        p = params_from_row(row)
        K, f_K = rnd_strike(p, row["F"], k)
        ax.plot(K, f_K, lw=1.8, label=f"{row['expiry']} (T={row['T']:.2f}y)")
    ax.set_xlabel("SPX level  $S_T$")
    ax.set_ylabel("risk-neutral density")
    ax.set_title("SPX market-implied risk-neutral density")
    ax.legend()
    ax.grid(alpha=0.3)
    d = cfg.paths.resolve("output_dir")
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path

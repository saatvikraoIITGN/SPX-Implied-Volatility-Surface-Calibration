"""Gatheral's raw-SVI parametrization, calibrated per expiry slice.

Raw SVI models total variance as a function of log-moneyness ``k``:

    w(k) = a + b · [ ρ·(k - m) + √((k - m)² + σ²) ]

Five parameters carry clear economic meaning:
    a      overall variance level (vertical shift)
    b ≥ 0  wing slope / overall convexity
    ρ      skew (rotation): for equity indices ρ < 0 (downside vols richer)
    m      horizontal shift of the smile minimum
    σ > 0  ATM curvature (how rounded the smile bottom is)

Each expiry slice is fit by vega-weighted least squares with multi-start initialization
for robustness. After fitting we evaluate Gatheral's ``g(k)`` to certify the slice is
butterfly-arbitrage free (non-negative risk-neutral density).

Reference: Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces" (2014).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .config import Config
from .iv_solver import bs_vega

SVI_PARAM_COLUMNS = [
    "expiry", "T", "F", "a", "b", "rho", "m", "sigma",
    "n_points", "rmse_vol", "butterfly_free",
]


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray | float) -> np.ndarray | float:
        x = np.asarray(k, dtype="float64") - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma * self.sigma))

    def implied_vol(self, k: np.ndarray | float, T: float) -> np.ndarray | float:
        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / T)

    def derivatives(self, k: np.ndarray | float):
        """Return (w, w', w'') of total variance w.r.t. k."""
        x = np.asarray(k, dtype="float64") - self.m
        R = np.sqrt(x * x + self.sigma * self.sigma)
        w = self.a + self.b * (self.rho * x + R)
        wp = self.b * (self.rho + x / R)
        wpp = self.b * self.sigma * self.sigma / (R ** 3)
        return w, wp, wpp

    def as_tuple(self) -> tuple:
        return (self.a, self.b, self.rho, self.m, self.sigma)


def g_function(params: SVIParams, k: np.ndarray) -> np.ndarray:
    """Gatheral's g(k); the risk-neutral density is non-negative iff g(k) >= 0."""
    w, wp, wpp = params.derivatives(k)
    w = np.maximum(w, 1e-12)
    term1 = (1.0 - 0.5 * k * wp / w) ** 2
    term2 = -0.25 * (wp ** 2) * (1.0 / w + 0.25)
    term3 = 0.5 * wpp
    return term1 + term2 + term3


def is_butterfly_free(params: SVIParams, k_grid: np.ndarray, tol: float = -1e-6) -> bool:
    return bool(np.all(g_function(params, k_grid) >= tol))


def _starts(k: np.ndarray, w: np.ndarray, n: int) -> list[tuple]:
    """Deterministic multi-start initial guesses for (a, b, rho, m, sigma)."""
    w_min = max(float(np.min(w)), 1e-6)
    m_atmin = float(k[int(np.argmin(w))])
    starts = []
    for a0 in (w_min, float(np.median(w))):
        for b0 in (0.1, 0.5):
            for rho0 in (-0.7, -0.3, 0.0):
                for m0 in (0.0, m_atmin):
                    for s0 in (0.1, 0.4):
                        starts.append((a0, b0, rho0, m0, s0))
    return starts[:max(1, n)]


def fit_slice(
    k: np.ndarray,
    w: np.ndarray,
    weights: np.ndarray | None = None,
    cfg: Config | None = None,
) -> tuple[SVIParams, float]:
    """Calibrate raw SVI to one slice of (log-moneyness, total variance).

    Returns ``(params, rmse_vol)`` where the RMSE is in volatility points (not variance).
    """
    cfg = cfg or Config.load()
    sc = cfg.svi
    k = np.asarray(k, dtype="float64")
    w = np.asarray(w, dtype="float64")
    if weights is None:
        weights = np.ones_like(w)
    sqrt_w = np.sqrt(np.maximum(weights, 0.0))

    spread_k = float(np.ptp(k)) or 1.0
    w_max = float(np.max(w))
    lb = np.array([0.0, sc.b_min, -sc.rho_abs_max, k.min() - spread_k, sc.sigma_min])
    ub = np.array([2.0 * w_max + 1e-6, 10.0, sc.rho_abs_max, k.max() + spread_k, 5.0])

    def residuals(theta):
        a, b, rho, m, sig = theta
        model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig * sig))
        return sqrt_w * (model - w)

    best, best_cost = None, np.inf
    for theta0 in _starts(k, w, sc.n_multistart):
        theta0 = np.clip(theta0, lb, ub)
        try:
            res = least_squares(residuals, theta0, bounds=(lb, ub), method="trf", max_nfev=2000)
        except Exception:  # noqa: BLE001 - a bad start shouldn't kill the calibration
            continue
        if res.cost < best_cost:
            best, best_cost = res.x, res.cost

    if best is None:  # all starts failed; fall back to a flat-variance fit
        best = np.array([max(float(np.median(w)), 1e-6), 0.0, 0.0, 0.0, 0.1])

    params = SVIParams(*[float(v) for v in best])
    return params, _rmse_w(params, k, w)


def _rmse_w(params: SVIParams, k: np.ndarray, w: np.ndarray) -> float:
    """RMSE of the fit in total-variance space."""
    model_w = np.maximum(params.total_variance(k), 0.0)
    return float(np.sqrt(np.mean((model_w - w) ** 2)))


def fit_surface(iv_frame: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Fit raw SVI per expiry slice; return one row of params per expiry.

    Expects columns ``expiry, T, F, log_moneyness, w, iv, strike, disc`` (output of the IV
    stage). Slices with too few points to identify 5 parameters are skipped.
    """
    cfg = cfg or Config.load()
    rows = []
    for expiry, sub in iv_frame.groupby("expiry"):
        s = sub.sort_values("log_moneyness")
        if len(s) < 5:
            continue
        T = float(s["T"].iloc[0])
        F = float(s["F"].iloc[0])
        k = s["log_moneyness"].to_numpy()
        w = s["w"].to_numpy()
        weights = _vega_weights(s)
        params, rmse_w = fit_slice(k, w, weights, cfg)
        rmse_vol = _slice_rmse_vol(params, k, s["iv"].to_numpy(), T)
        free = is_butterfly_free(params, np.linspace(k.min() - 0.1, k.max() + 0.1, 200))
        rows.append({
            "expiry": expiry, "T": T, "F": F,
            "a": params.a, "b": params.b, "rho": params.rho,
            "m": params.m, "sigma": params.sigma,
            "n_points": len(s), "rmse_vol": rmse_vol, "butterfly_free": free,
        })
    return pd.DataFrame(rows, columns=SVI_PARAM_COLUMNS).sort_values("T").reset_index(drop=True)


def _slice_rmse_vol(params: SVIParams, k: np.ndarray, iv_mkt: np.ndarray, T: float) -> float:
    iv_model = params.implied_vol(k, T)
    return float(np.sqrt(np.mean((iv_model - iv_mkt) ** 2)))


def _vega_weights(s: pd.DataFrame) -> np.ndarray:
    """Vega weights (emphasize ATM, where quotes are most informative)."""
    if not {"F", "disc", "strike", "T", "iv"}.issubset(s.columns):
        return np.ones(len(s))
    vega = np.array([
        bs_vega(r.F, r.strike, r.T, r.iv, r.disc) for r in s.itertuples(index=False)
    ])
    vega = np.nan_to_num(vega, nan=0.0)
    if vega.max() <= 0:
        return np.ones(len(s))
    return vega / vega.max()


def params_from_row(row: pd.Series) -> SVIParams:
    """Reconstruct SVIParams from a fit_surface row."""
    return SVIParams(a=row["a"], b=row["b"], rho=row["rho"], m=row["m"], sigma=row["sigma"])

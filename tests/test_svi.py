"""Tests for raw-SVI calibration and the butterfly-free certificate."""
from datetime import date

import numpy as np
import pandas as pd

from volsurface.svi import (
    SVIParams,
    fit_slice,
    fit_surface,
    g_function,
    is_butterfly_free,
)

TRUE = SVIParams(a=0.04, b=0.4, rho=-0.3, m=0.0, sigma=0.2)
KGRID = np.linspace(-0.5, 0.5, 41)


def test_total_variance_and_derivatives_consistent():
    k = KGRID
    w, wp, wpp = TRUE.derivatives(k)
    # Finite-difference check of the analytic derivatives.
    dk = 1e-5
    wp_fd = (TRUE.total_variance(k + dk) - TRUE.total_variance(k - dk)) / (2 * dk)
    assert np.allclose(wp, wp_fd, atol=1e-4)


def test_recovers_curve_from_clean_data():
    w = np.asarray(TRUE.total_variance(KGRID))
    fitted, rmse_w = fit_slice(KGRID, w)
    assert np.allclose(fitted.total_variance(KGRID), w, atol=1e-3)
    assert rmse_w < 1e-3


def test_recovers_curve_with_small_noise():
    rng = np.random.default_rng(0)
    w = np.asarray(TRUE.total_variance(KGRID)) + rng.normal(0, 1e-4, size=KGRID.size)
    fitted, _ = fit_slice(KGRID, w)
    assert np.allclose(fitted.total_variance(KGRID), TRUE.total_variance(KGRID), atol=5e-3)


def test_wellbehaved_slice_is_butterfly_free():
    assert is_butterfly_free(TRUE, np.linspace(-1.0, 1.0, 400))


def test_pathological_slice_violates_butterfly():
    # Steep wings + near-degenerate curvature -> negative density somewhere.
    bad = SVIParams(a=0.001, b=1.2, rho=-0.95, m=0.0, sigma=0.02)
    g = g_function(bad, np.linspace(-1.0, 1.0, 400))
    assert g.min() < 0
    assert not is_butterfly_free(bad, np.linspace(-1.0, 1.0, 400))


def test_fit_surface_two_expiries():
    rows = []
    for expiry, T in [(date(2026, 9, 18), 0.5), (date(2026, 12, 18), 1.0)]:
        for k in KGRID:
            w = float(TRUE.total_variance(k)) * (T / 0.5)  # scale variance with maturity
            iv = (w / T) ** 0.5
            rows.append({
                "expiry": expiry, "T": T, "F": 5000.0,
                "log_moneyness": float(k), "w": w, "iv": iv,
                "strike": 5000.0 * np.exp(k), "disc": 0.98,
            })
    surf = fit_surface(pd.DataFrame(rows))
    assert len(surf) == 2
    assert surf["butterfly_free"].all()
    assert (surf["rmse_vol"] < 1e-2).all()
    assert set(["a", "b", "rho", "m", "sigma"]).issubset(surf.columns)

"""Tests for surface assembly and plotting (numeric + smoke)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from volsurface.config import Config
from volsurface.surface import (
    atm_term_structure,
    evaluate_surface,
    plot_atm_term_structure,
    plot_smile,
    plot_surface_3d,
)

A, B, RHO, M, SIG = 0.04, 0.4, -0.3, 0.0, 0.2


def _svi_df():
    rows = []
    for expiry, T in [(date(2026, 9, 18), 0.5), (date(2026, 12, 18), 1.0)]:
        rows.append({
            "expiry": expiry, "T": T, "F": 5000.0,
            "a": A * (T / 0.5), "b": B, "rho": RHO, "m": M, "sigma": SIG,
            "n_points": 30, "rmse_vol": 0.004, "butterfly_free": True,
        })
    return pd.DataFrame(rows)


def _iv_points(expiry=date(2026, 9, 18), T=0.5):
    from volsurface.iv_solver import bs_price
    from volsurface.svi import SVIParams
    p = SVIParams(A, B, RHO, M, SIG)
    F, disc = 5000.0, 0.98
    rows = []
    for k in np.linspace(-0.2, 0.1, 15):
        K = F * np.exp(k)
        iv = float(p.implied_vol(k, T))
        typ = "C" if K >= F else "P"
        mid = bs_price(F, K, T, iv, disc, typ)
        rows.append({
            "expiry": expiry, "type": typ, "strike": K, "F": F, "disc": disc,
            "T": T, "log_moneyness": k, "iv": iv, "w": iv**2 * T,
            "mid": mid, "bid": mid * 0.99, "ask": mid * 1.01,
        })
    return pd.DataFrame(rows)


def test_evaluate_surface_shape_and_positive():
    grid = evaluate_surface(_svi_df(), k_grid=np.linspace(-0.2, 0.1, 25))
    assert grid.iv.shape == (2, 25)
    assert np.all(grid.iv > 0)


def test_atm_term_structure_value():
    ts = atm_term_structure(_svi_df())
    # T=0.5 slice: w0 = a + b*sigma = 0.04 + 0.4*0.2 = 0.12 -> atm_iv = sqrt(0.12/0.5)
    atm05 = ts[ts["T"] == 0.5]["atm_iv"].iloc[0]
    assert atm05 == pytest.approx(np.sqrt(0.12 / 0.5))


def test_plots_write_files(tmp_path):
    cfg = Config.load()
    object.__setattr__(cfg.paths, "output_dir", str(tmp_path))
    svi_df = _svi_df()
    p1 = plot_surface_3d(svi_df, cfg)
    p2 = plot_atm_term_structure(svi_df, cfg)
    p3 = plot_smile(svi_df, _iv_points(), date(2026, 9, 18), cfg)
    assert p1.exists() and p2.exists() and p3.exists()
    assert p1.stat().st_size > 0

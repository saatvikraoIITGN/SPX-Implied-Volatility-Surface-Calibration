"""Tests for the SVI-implied risk-neutral density."""
from datetime import date

import numpy as np
import pandas as pd

from volsurface.config import Config
from volsurface.density import (
    density_stats,
    plot_rnd,
    rnd_logmoneyness,
    rnd_strike,
)
from volsurface.svi import SVIParams

# Realistic 1Y-ish slice; raw-SVI linear wings stay light enough that the density
# integrates to ~1 on a sane grid (very large b creates heavy tails needing a wider grid).
TRUE = SVIParams(a=0.04, b=0.15, rho=-0.4, m=0.0, sigma=0.15)  # butterfly-free


def test_density_integrates_to_one():
    k = np.linspace(-4.0, 4.0, 8000)
    stats = density_stats(TRUE, k)
    assert abs(stats.total_prob - 1.0) < 1e-3


def test_density_non_negative_for_arbfree_slice():
    k = np.linspace(-2.0, 2.0, 2000)
    p = rnd_logmoneyness(TRUE, k)
    assert p.min() >= -1e-12


def test_strike_density_positive_and_located_near_forward():
    F = 5000.0
    k = np.linspace(-1.0, 1.0, 2000)
    K, f_K = rnd_strike(TRUE, F, k)
    assert np.all(f_K >= -1e-12)
    # Mode of the distribution should sit in a sensible band around the forward.
    mode_K = K[int(np.argmax(f_K))]
    assert 0.6 * F < mode_K < 1.2 * F


def test_plot_rnd_writes_file(tmp_path):
    cfg = Config.load()
    object.__setattr__(cfg.paths, "output_dir", str(tmp_path))
    rows = []
    for expiry, T in [(date(2026, 9, 18), 0.5), (date(2026, 12, 18), 1.0)]:
        rows.append({"expiry": expiry, "T": T, "F": 5000.0,
                     "a": 0.04 * (T / 0.5), "b": 0.15, "rho": -0.4, "m": 0.0,
                     "sigma": 0.15, "rmse_vol": 0.004, "butterfly_free": True})
    path = plot_rnd(pd.DataFrame(rows), cfg=cfg)
    assert path.exists() and path.stat().st_size > 0

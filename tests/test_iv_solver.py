"""Tests for the Black-76 pricer and implied-vol inversion."""
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from volsurface.iv_solver import (
    bs_price,
    bs_vega,
    implied_vol,
    select_otm,
    solve_iv_frame,
)

F, T, DISC = 5000.0, 0.5, 0.98


@pytest.mark.parametrize("opt_type", ["C", "P"])
@pytest.mark.parametrize("K", [4000.0, 4800.0, 5000.0, 5200.0, 6000.0])
@pytest.mark.parametrize("sigma", [0.08, 0.20, 0.45])
def test_price_iv_roundtrip(opt_type, K, sigma):
    price = bs_price(F, K, T, sigma, DISC, opt_type)
    recovered = implied_vol(price, F, K, T, DISC, opt_type)
    assert recovered == pytest.approx(sigma, abs=1e-5)


def test_put_call_parity_in_pricer():
    K = 5100.0
    c = bs_price(F, K, T, 0.2, DISC, "C")
    p = bs_price(F, K, T, 0.2, DISC, "P")
    assert (c - p) == pytest.approx(DISC * (F - K), abs=1e-8)


def test_vega_positive_and_peaks_atm():
    atm = bs_vega(F, F, T, 0.2, DISC)
    wing = bs_vega(F, 7000.0, T, 0.2, DISC)
    assert atm > 0
    assert atm > wing  # vega is largest near the money


def test_iv_nan_when_price_below_intrinsic():
    K = 4000.0  # deep ITM call; intrinsic (fwd) = F-K = 1000
    below = DISC * (F - K) * 0.5  # below intrinsic -> no BS vol
    assert math.isnan(implied_vol(below, F, K, T, DISC, "C"))


def test_iv_nan_when_price_above_upper_bound():
    K = 5000.0
    too_high = DISC * F * 1.01  # call value can't exceed discounted forward
    assert math.isnan(implied_vol(too_high, F, K, T, DISC, "C"))


def test_select_otm_picks_correct_leg():
    rows = []
    for k in (4500.0, 5000.0, 5500.0):
        for typ in ("C", "P"):
            rows.append({"expiry": date(2026, 9, 18), "type": typ, "strike": k,
                         "F": 5000.0, "T": T, "mid": 50.0})
    df = pd.DataFrame(rows)
    otm = select_otm(df)
    # K<F -> put, K>=F -> call
    assert otm[otm.strike == 4500.0]["type"].iloc[0] == "P"
    assert otm[otm.strike == 5000.0]["type"].iloc[0] == "C"
    assert otm[otm.strike == 5500.0]["type"].iloc[0] == "C"
    assert len(otm) == 3


def test_solve_iv_frame_adds_iv_and_w():
    sigma_true = 0.22
    rows = []
    for k in (4600.0, 4800.0, 5000.0, 5200.0, 5400.0):
        typ = "C" if k >= F else "P"
        price = bs_price(F, k, T, sigma_true, DISC, typ)
        rows.append({"expiry": date(2026, 9, 18), "type": typ, "strike": k,
                     "F": F, "disc": DISC, "T": T, "mid": price})
    out = solve_iv_frame(pd.DataFrame(rows))
    assert len(out) == 5
    assert np.allclose(out["iv"], sigma_true, atol=1e-5)
    assert np.allclose(out["w"], sigma_true**2 * T, atol=1e-6)

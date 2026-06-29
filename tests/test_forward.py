"""Tests for forward & discount extraction via put-call parity."""
import math
from datetime import date

import numpy as np
import pandas as pd

from volsurface.forward import attach_forward, extract_forwards, parity_frame
from volsurface.sources.base import RawChain, normalize_quotes

ASOF = date(2026, 6, 29)


def _parity_slice(F, r, T, spot, strikes, expiry=date(2026, 12, 18)):
    """Build a cleaned-style slice whose C-P satisfies parity exactly for (F, r)."""
    disc = math.exp(-r * T)
    rows = []
    base = 400.0
    for k in strikes:
        g = disc * (F - k)            # C - P must equal this
        c_mid = base + 0.5 * g
        p_mid = base - 0.5 * g
        rows.append({"expiry": expiry, "type": "C", "strike": float(k),
                     "mid": c_mid, "T": T, "spot": spot})
        rows.append({"expiry": expiry, "type": "P", "strike": float(k),
                     "mid": p_mid, "T": T, "spot": spot})
    return pd.DataFrame(rows)


def test_recovers_known_forward_and_rate():
    F, r, T, spot = 5050.0, 0.045, 0.5, 5000.0
    strikes = np.arange(4600, 5401, 50)
    slice_df = _parity_slice(F, r, T, spot, strikes)

    fwd = extract_forwards(slice_df)
    assert len(fwd) == 1
    row = fwd.iloc[0]
    assert row["method"] == "parity"
    assert abs(row["F"] - F) < 1e-6
    assert abs(row["disc"] - math.exp(-r * T)) < 1e-9
    assert abs(row["rate"] - r) < 1e-9
    # Market-implied forward differs from naive spot, as intended.
    assert abs(row["F"] - spot) > 1.0


def test_log_moneyness_attached():
    F, r, T, spot = 5050.0, 0.045, 0.5, 5000.0
    strikes = np.arange(4600, 5401, 50)
    slice_df = _parity_slice(F, r, T, spot, strikes)
    fwd = extract_forwards(slice_df)
    attached = attach_forward(slice_df, fwd)
    assert "log_moneyness" in attached.columns
    k5000 = attached[(attached["strike"] == 5000.0) & (attached["type"] == "C")].iloc[0]
    assert abs(k5000["log_moneyness"] - math.log(5000.0 / F)) < 1e-9


def test_parity_frame_selects_both_legs_no_oi_gate():
    """parity_frame keeps quotes regardless of OI, within tenor/moneyness/spread limits."""
    spot = 5000.0
    rows = []
    for k in (4600, 5000, 5400):
        for typ in ("C", "P"):
            rows.append({
                "expiry": date(2026, 9, 18), "type": typ, "strike": float(k),
                "bid": 10.0, "ask": 10.5, "last": 10.2,
                "volume": 0, "open_interest": 0,  # zero OI must NOT exclude it here
                "spot": spot,
            })
    # Out-of-band strike (moneyness 1.5) should be excluded.
    rows.append({"expiry": date(2026, 9, 18), "type": "C", "strike": 7500.0,
                 "bid": 1.0, "ask": 1.2, "last": 1.1, "volume": 0,
                 "open_interest": 0, "spot": spot})
    chain = RawChain(quotes=normalize_quotes(pd.DataFrame(rows)), asof=ASOF,
                     spot=spot, ticker="^SPX")
    pf = parity_frame(chain)
    assert set(pf.columns) == {"expiry", "type", "strike", "mid", "T", "spot"}
    assert sorted(pf["strike"].unique()) == [4600.0, 5000.0, 5400.0]
    assert (pf["mid"] == 10.25).all()


def test_fallback_when_too_few_pairs():
    F, r, T, spot = 5050.0, 0.045, 0.5, 5000.0
    slice_df = _parity_slice(F, r, T, spot, strikes=[5000.0])  # only 1 pair
    fwd = extract_forwards(slice_df)
    row = fwd.iloc[0]
    assert row["method"] == "fallback"
    # Fallback forward uses configured carry: F = spot / disc.
    assert abs(row["F"] - spot / row["disc"]) < 1e-9

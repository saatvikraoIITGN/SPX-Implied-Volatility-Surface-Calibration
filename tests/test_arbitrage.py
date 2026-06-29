"""Tests for calendar and butterfly no-arbitrage checks."""
import math
from datetime import date

import pandas as pd

from volsurface.arbitrage import (
    drop_butterfly_violations,
    flag_butterfly,
    flag_calendar,
    undiscounted_call,
)

EXP = date(2026, 9, 18)


def _smile(strikes, call_prices, F=5000.0, T=0.5):
    """One expiry, all calls, disc=1 so undiscounted call price == mid."""
    return pd.DataFrame({
        "expiry": EXP, "type": "C", "strike": [float(k) for k in strikes],
        "mid": [float(c) for c in call_prices], "disc": 1.0, "F": F, "T": T,
        "log_moneyness": [math.log(k / F) for k in strikes],
        "w": 0.04 * T,
    })


def test_undiscounted_call_uses_parity_for_puts():
    df = pd.DataFrame({
        "type": ["C", "P"], "strike": [5000.0, 4800.0], "mid": [100.0, 30.0],
        "disc": [0.98, 0.98], "F": [5050.0, 5050.0],
    })
    cfwd = undiscounted_call(df)
    assert cfwd[0] == 100.0 / 0.98                      # call: mid/disc
    assert cfwd[1] == 30.0 / 0.98 + (5050.0 - 4800.0)   # put -> call via parity


def test_butterfly_clean_smile_has_no_violation():
    df = _smile([4800, 4900, 5000, 5100, 5200], [250, 170, 100, 55, 25])
    flagged = flag_butterfly(df)
    assert flagged["butterfly_ok"].all()


def test_butterfly_detects_concave_kink():
    # Middle price bumped up -> call price non-convex at strike 5000.
    df = _smile([4800, 4900, 5000, 5100, 5200], [250, 170, 200, 55, 25])
    flagged = flag_butterfly(df)
    bad = flagged[~flagged["butterfly_ok"]]
    assert len(bad) == 1
    assert bad["strike"].iloc[0] == 5000.0
    assert len(drop_butterfly_violations(flagged)) == 4


def _cal_slice(expiry, T, ks, ws):
    return pd.DataFrame({
        "expiry": expiry, "T": T, "log_moneyness": ks, "w": ws,
    })


def test_calendar_detects_decreasing_total_variance():
    a = _cal_slice(date(2026, 9, 18), 0.25, [-0.1, 0.0, 0.1], [0.04, 0.03, 0.04])
    b = _cal_slice(date(2026, 12, 18), 0.50, [-0.1, 0.0, 0.1], [0.05, 0.02, 0.05])
    frame = pd.concat([a, b], ignore_index=True)
    per_expiry, total = flag_calendar(frame)
    assert total > 0
    assert date(2026, 12, 18) in per_expiry


def test_calendar_clean_when_variance_increases():
    a = _cal_slice(date(2026, 9, 18), 0.25, [-0.1, 0.0, 0.1], [0.03, 0.02, 0.03])
    b = _cal_slice(date(2026, 12, 18), 0.50, [-0.1, 0.0, 0.1], [0.06, 0.05, 0.06])
    frame = pd.concat([a, b], ignore_index=True)
    _, total = flag_calendar(frame)
    assert total == 0


def test_calendar_no_overlap_returns_zero():
    a = _cal_slice(date(2026, 9, 18), 0.25, [-0.2, -0.15], [0.04, 0.035])
    b = _cal_slice(date(2026, 12, 18), 0.50, [0.15, 0.2], [0.05, 0.055])
    frame = pd.concat([a, b], ignore_index=True)
    _, total = flag_calendar(frame)
    assert total == 0

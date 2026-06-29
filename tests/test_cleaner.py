"""Tests for cleaning/filtering with a deterministic synthetic dirty chain."""
from dataclasses import replace
from datetime import date

import pandas as pd

from volsurface.cleaner import CLEANED_COLUMNS, clean_chain
from volsurface.config import Config
from volsurface.sources.base import RawChain, normalize_quotes

ASOF = date(2026, 6, 29)
SPOT = 5000.0


def _cfg():
    """Pin cleaning thresholds so the test is independent of shipped defaults."""
    base = Config.load()
    cleaning = replace(
        base.cleaning,
        min_open_interest=10,
        min_volume=0,
        max_rel_spread=1.5,
        moneyness_min=0.7,
        moneyness_max=1.3,
        min_strikes_per_expiry=6,
    )
    return replace(base, cleaning=cleaning)


def _row(expiry, strike, bid, ask, oi=100, typ="C", vol=10, last=None):
    return {
        "expiry": expiry, "type": typ, "strike": float(strike),
        "bid": float(bid), "ask": float(ask),
        "last": float(last if last is not None else 0.5 * (bid + ask)),
        "volume": int(vol), "open_interest": int(oi), "spot": SPOT,
    }


def _dirty_chain():
    exp_a = date(2026, 9, 18)   # ~81d, in window
    exp_b = date(2026, 12, 18)  # in window but thin (only 3 strikes)
    rows = []
    # 7 clean strikes on expiry A -> all survive
    for k in (4600, 4700, 4800, 4900, 5000, 5100, 5200):
        rows.append(_row(exp_a, k, bid=k * 0.02, ask=k * 0.02 + 2))
    # dirty rows on expiry A
    rows.append(_row(exp_a, 4550, bid=0.0, ask=5.0))          # non_positive_bid
    rows.append(_row(exp_a, 4560, bid=10.0, ask=8.0))         # crossed_market
    rows.append(_row(exp_a, 4570, bid=5.0, ask=6.0, oi=2))    # low_open_interest
    rows.append(_row(exp_a, 4580, bid=0.1, ask=5.0))          # wide_spread (rel~1.9)
    rows.append(_row(exp_a, 7000, bid=1.0, ask=2.0))          # moneyness_out_of_range
    # thin expiry B: 3 clean strikes -> whole slice dropped
    for k in (4900, 5000, 5100):
        rows.append(_row(exp_b, k, bid=k * 0.015, ask=k * 0.015 + 2))
    # tenor filters
    rows.append(_row(date(2026, 6, 20), 5000, bid=50, ask=52))  # non_positive_dte (past)
    rows.append(_row(date(2026, 7, 2), 5000, bid=20, ask=22))   # dte_out_of_window (<7d)

    quotes = normalize_quotes(pd.DataFrame(rows))
    return RawChain(quotes=quotes, asof=ASOF, spot=SPOT, ticker="^SPX")


def test_cleaning_survivors_and_report():
    cfg = _cfg()
    cleaned, report = clean_chain(_dirty_chain(), cfg)

    assert list(cleaned.columns) == CLEANED_COLUMNS
    assert report.n_input == 17
    assert report.n_output == 7
    assert len(cleaned) == 7

    # Only expiry A's 7 clean strikes survive.
    assert cleaned["expiry"].nunique() == 1
    assert sorted(cleaned["strike"]) == [4600, 4700, 4800, 4900, 5000, 5100, 5200]

    expected = {
        "non_positive_dte": 1,
        "dte_out_of_window": 1,
        "non_positive_bid": 1,
        "crossed_market": 1,
        "low_open_interest": 1,
        "wide_spread": 1,
        "moneyness_out_of_range": 1,
        "thin_expiry_slice": 3,
    }
    assert report.dropped == expected
    assert report.expiries_dropped == 1


def test_derived_columns_are_correct():
    cfg = _cfg()
    cleaned, _ = clean_chain(_dirty_chain(), cfg)
    row = cleaned[cleaned["strike"] == 5000.0].iloc[0]
    assert row["mid"] == 0.5 * (row["bid"] + row["ask"])
    assert row["spread"] == row["ask"] - row["bid"]
    assert abs(row["moneyness"] - 5000.0 / SPOT) < 1e-12
    assert row["T"] > 0


def test_report_to_frame_sums_to_input():
    cfg = _cfg()
    _, report = clean_chain(_dirty_chain(), cfg)
    frame = report.to_frame()
    assert int(frame["count"].sum()) == report.n_input

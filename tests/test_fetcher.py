"""Tests for the source-agnostic fetcher and raw-chain contract (no network)."""
from datetime import date

import pandas as pd
import pytest

from volsurface.config import Config
from volsurface.fetcher import fetch_chain, load_raw, save_raw
from volsurface.sources.base import RAW_COLUMNS, ChainSource, RawChain, normalize_quotes


def _raw_df():
    return pd.DataFrame(
        {
            "expiry": [date(2026, 7, 17), date(2026, 7, 17)],
            "type": ["C", "P"],
            "strike": [5000.0, 5000.0],
            "bid": [120.0, 95.0],
            "ask": [122.0, 97.0],
            "last": [121.0, 96.0],
            "volume": [10, 12],
            "open_interest": [500, 600],
            "spot": [5025.0, 5025.0],
        }
    )


class FakeSource(ChainSource):
    def __init__(self, df):
        self._df = df

    def fetch(self, asof=None):
        return RawChain(
            quotes=normalize_quotes(self._df),
            asof=asof or date(2026, 6, 29),
            spot=5025.0,
            ticker="^SPX",
        )


def test_rawchain_requires_all_columns():
    bad = _raw_df().drop(columns=["bid"])
    with pytest.raises(ValueError, match="missing columns"):
        RawChain(quotes=bad, asof=date(2026, 6, 29), spot=1.0, ticker="^SPX")


def test_rawchain_rejects_bad_option_type():
    df = normalize_quotes(_raw_df())
    df.loc[0, "type"] = "X"
    with pytest.raises(ValueError, match="invalid option types"):
        RawChain(quotes=df, asof=date(2026, 6, 29), spot=1.0, ticker="^SPX")


def test_normalize_dtypes_and_order():
    out = normalize_quotes(_raw_df())
    assert list(out.columns) == RAW_COLUMNS
    assert out["strike"].dtype == "float64"
    assert out["open_interest"].dtype == "int64"
    assert set(out["type"]) <= {"C", "P"}


def test_fetch_with_injected_source_no_persist():
    chain = fetch_chain(source=FakeSource(_raw_df()), persist=False)
    assert chain.n_quotes == 2
    assert chain.ticker == "^SPX"
    assert chain.expiries == [date(2026, 7, 17)]


def test_persist_and_reload_roundtrip(tmp_path):
    cfg = Config.load()
    # Redirect the raw dir into a temp location for the test.
    object.__setattr__(cfg.paths, "raw_dir", str(tmp_path))
    chain = FakeSource(_raw_df()).fetch()
    path = save_raw(chain, cfg)
    assert path.exists()
    reloaded = load_raw(path)
    assert reloaded.asof == chain.asof
    assert reloaded.ticker == chain.ticker
    assert reloaded.spot == chain.spot
    pd.testing.assert_frame_equal(reloaded.quotes, chain.quotes)

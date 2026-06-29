"""The source-agnostic seam for raw option-chain data.

Every concrete data provider (yfinance, a historical EOD dump, a paid vendor) implements
``ChainSource.fetch`` and returns a :class:`RawChain`: a normalized long-format frame plus
snapshot metadata. Downstream pipeline stages depend only on this contract, never on the
provider, so swapping the source touches nothing else.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import date

import pandas as pd

# Canonical raw-chain columns. One row per (expiry, type, strike) quote.
RAW_COLUMNS = [
    "expiry",         # option expiration date (datetime.date)
    "type",           # "C" or "P"
    "strike",         # strike price (float)
    "bid",            # best bid (float, may be 0/NaN)
    "ask",            # best ask (float, may be 0/NaN)
    "last",           # last traded price (float, may be NaN)
    "volume",         # contracts traded today (int, may be 0)
    "open_interest",  # open interest (int, may be 0)
    "spot",           # underlying spot at snapshot time (float, constant per snapshot)
]

OPTION_TYPES = ("C", "P")


@dataclass(frozen=True)
class RawChain:
    """A single point-in-time option-chain snapshot.

    Attributes
    ----------
    quotes:
        Long-format DataFrame with exactly :data:`RAW_COLUMNS`.
    asof:
        Snapshot date.
    spot:
        Underlying spot price at snapshot time.
    ticker:
        Underlying ticker the chain was pulled for (e.g. ``^SPX``).
    """

    quotes: pd.DataFrame
    asof: date
    spot: float
    ticker: str

    def __post_init__(self) -> None:
        missing = set(RAW_COLUMNS) - set(self.quotes.columns)
        if missing:
            raise ValueError(f"RawChain.quotes missing columns: {sorted(missing)}")
        bad = set(self.quotes["type"].unique()) - set(OPTION_TYPES)
        if bad:
            raise ValueError(f"RawChain.quotes has invalid option types: {sorted(bad)}")

    @property
    def n_quotes(self) -> int:
        return len(self.quotes)

    @property
    def expiries(self) -> list[date]:
        return sorted(self.quotes["expiry"].unique())


class ChainSource(abc.ABC):
    """Abstract provider of raw option chains."""

    @abc.abstractmethod
    def fetch(self, asof: date | None = None) -> RawChain:
        """Fetch the option chain.

        Parameters
        ----------
        asof:
            Requested snapshot date. Live sources may ignore it (they only know "now")
            and stamp the actual snapshot date on the returned :class:`RawChain`.
        """
        raise NotImplementedError


def normalize_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a provider frame to the canonical raw schema, dtypes, and column order.

    Extra columns are dropped; required columns must already be present.
    """
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"cannot normalize, missing columns: {sorted(missing)}")
    out = df.loc[:, RAW_COLUMNS].copy()
    for col in ("strike", "bid", "ask", "last", "spot"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in ("volume", "open_interest"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int64")
    out["type"] = out["type"].astype(str).str.upper().str[0]
    return out.reset_index(drop=True)

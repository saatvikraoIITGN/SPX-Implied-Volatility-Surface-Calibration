"""Fetch and persist a raw SPX option-chain snapshot.

Thin orchestration over a :class:`ChainSource`: pull the chain, persist it to
``data/raw/spx_<asof>.parquet``, and return the :class:`RawChain`. The default source is
yfinance, but any ``ChainSource`` can be injected (e.g. in tests or for a historical feed).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .config import Config
from .sources.base import ChainSource, RawChain
from .sources.yfinance_source import YFinanceSource


def default_source(cfg: Config) -> ChainSource:
    """Build the configured default source (yfinance)."""
    return YFinanceSource(
        ticker=cfg.underlying.ticker,
        fallback_tickers=list(cfg.underlying.fallback_tickers),
    )


def fetch_chain(
    cfg: Config | None = None,
    source: ChainSource | None = None,
    asof: date | None = None,
    persist: bool = True,
) -> RawChain:
    """Fetch a raw chain and (optionally) persist it.

    Parameters
    ----------
    cfg:
        Pipeline config; defaults to :meth:`Config.load`.
    source:
        Override the data source. Defaults to the configured yfinance source.
    asof:
        Requested snapshot date (live sources stamp the actual date).
    persist:
        If True, write the snapshot parquet under ``cfg.paths.raw_dir``.
    """
    cfg = cfg or Config.load()
    source = source or default_source(cfg)
    chain = source.fetch(asof=asof)
    if persist:
        save_raw(chain, cfg)
    return chain


def raw_path(cfg: Config, asof: date) -> Path:
    raw_dir = cfg.paths.resolve("raw_dir")
    return raw_dir / f"spx_{asof.isoformat()}.parquet"


def save_raw(chain: RawChain, cfg: Config) -> Path:
    """Persist a RawChain's quotes (plus snapshot metadata) to parquet."""
    out = raw_path(cfg, chain.asof)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = chain.quotes.copy()
    # Stash snapshot-level metadata as columns so the parquet is self-describing.
    df["asof"] = chain.asof.isoformat()
    df["ticker"] = chain.ticker
    df.to_parquet(out, index=False)
    return out


def load_raw(path: str | Path) -> RawChain:
    """Reload a persisted snapshot back into a :class:`RawChain`."""
    df = pd.read_parquet(path)
    asof = date.fromisoformat(str(df["asof"].iloc[0]))
    ticker = str(df["ticker"].iloc[0])
    spot = float(df["spot"].iloc[0])
    quotes = df.drop(columns=["asof", "ticker"])
    return RawChain(quotes=quotes, asof=asof, spot=spot, ticker=ticker)

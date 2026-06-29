"""yfinance-backed :class:`ChainSource` for SPX index options.

yfinance only exposes a *current* snapshot of the chain (no history). That is exactly
what the static-surface project needs. The network/parsing details are quarantined here;
everything downstream sees only the normalized :class:`RawChain`.
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from .base import ChainSource, RawChain, normalize_quotes


class YFinanceSource(ChainSource):
    """Pull an SPX option chain from yfinance.

    Parameters
    ----------
    ticker:
        Primary underlying ticker. ``^SPX`` (European, cash-settled) is the default.
    fallback_tickers:
        Tried in order if the primary returns no expiries (e.g. ``^SPXW``, ``SPY``).
    """

    def __init__(
        self,
        ticker: str = "^SPX",
        fallback_tickers: list[str] | None = None,
    ) -> None:
        self.ticker = ticker
        self.fallback_tickers = fallback_tickers or []

    def fetch(self, asof: date | None = None) -> RawChain:
        import yfinance as yf  # imported lazily so the package imports without network deps

        for tkr in [self.ticker, *self.fallback_tickers]:
            yt = yf.Ticker(tkr)
            expiries = tuple(yt.options or ())
            if not expiries:
                continue
            spot = _spot(yt)
            frames = []
            for exp_str in expiries:
                try:
                    chain = yt.option_chain(exp_str)
                except Exception:  # noqa: BLE001 - skip an expiry yfinance can't return
                    continue
                exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
                frames.append(_side(chain.calls, exp, "C", spot))
                frames.append(_side(chain.puts, exp, "P", spot))
            if not frames:
                continue
            quotes = normalize_quotes(pd.concat(frames, ignore_index=True))
            return RawChain(
                quotes=quotes,
                asof=asof or date.today(),
                spot=float(spot),
                ticker=tkr,
            )

        raise RuntimeError(
            f"no option expiries returned for {self.ticker} or fallbacks "
            f"{self.fallback_tickers}"
        )


def _spot(yt) -> float:
    """Best-effort spot from fast_info, falling back to the last daily close."""
    try:
        px = yt.fast_info["lastPrice"]
        if px and px > 0:
            return float(px)
    except Exception:  # noqa: BLE001
        pass
    hist = yt.history(period="1d")
    if len(hist):
        return float(hist["Close"].iloc[-1])
    raise RuntimeError("could not determine spot price")


def _side(df: pd.DataFrame, expiry: date, opt_type: str, spot: float) -> pd.DataFrame:
    """Map one yfinance calls/puts frame onto the canonical raw columns."""
    if df is None or len(df) == 0:
        return pd.DataFrame(
            columns=["expiry", "type", "strike", "bid", "ask", "last", "volume",
                     "open_interest", "spot"]
        )
    return pd.DataFrame(
        {
            "expiry": expiry,
            "type": opt_type,
            "strike": df["strike"].to_numpy(),
            "bid": df["bid"].to_numpy(),
            "ask": df["ask"].to_numpy(),
            "last": df.get("lastPrice", pd.Series(index=df.index, dtype="float64")).to_numpy(),
            "volume": df.get("volume", pd.Series(index=df.index, dtype="float64")).to_numpy(),
            "open_interest": df.get(
                "openInterest", pd.Series(index=df.index, dtype="float64")
            ).to_numpy(),
            "spot": spot,
        }
    )

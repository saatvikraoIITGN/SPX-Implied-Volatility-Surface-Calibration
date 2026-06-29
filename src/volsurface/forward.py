"""Per-expiry forward price and discount factor from put-call parity.

Put-call parity (European options) reads, per expiry T:

    C(K) - P(K) = D(T) * (F(T) - K),   with  D(T) = e^{-r T}.

So regressing the call-minus-put mid on strike K gives a line whose slope is ``-D`` and
intercept is ``D*F``. From the fitted ``(slope, intercept)`` we recover the **market-implied**
discount factor and forward — capturing dividends/carry without assuming ``F = S e^{rT}``.

This is the right way to anchor the smile: log-moneyness ``k = log(K/F)`` then uses the
forward the market is actually pricing, not a hand-set carry assumption.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .sources.base import RawChain

FORWARD_COLUMNS = ["expiry", "T", "F", "disc", "rate", "n_pairs", "method"]
# ATM bands (in |K/S - 1|) tried in order when selecting strikes for the parity regression.
# Parity holds at every strike, so a moderately wide band with many points is more robust
# than a tight one with a handful of noisy near-ATM pairs.
_ATM_BANDS = (0.10, 0.20, 0.35, math.inf)
_MIN_PAIRS = 5
# Sanity bound on the implied continuously-compounded rate; outside this, treat the fit
# as corrupted by stale/illiquid quotes and fall back.
_RATE_BOUNDS = (-0.10, 0.50)
# Light, OI-independent filters used only to select strikes for the parity regression.
_PARITY_MONEYNESS = (0.80, 1.20)
_PARITY_MAX_REL_SPREAD = 0.50


@dataclass
class ForwardResult:
    forwards: pd.DataFrame  # one row per expiry, columns = FORWARD_COLUMNS
    quotes: pd.DataFrame    # cleaned quotes + F, disc, log_moneyness


def parity_frame(chain: RawChain, cfg: Config | None = None) -> pd.DataFrame:
    """Liquidity-light, OI-independent selection of quotes for the parity regression.

    The OI threshold used to pick *smile* points is too aggressive for forward extraction:
    it usually kills one leg, so call/put strikes stop coinciding. Forward extraction
    instead needs both legs present, so it filters only for quote quality (positive,
    uncrossed, not absurdly wide) within a parity moneyness band — no OI gate.
    """
    cfg = cfg or Config.load()
    q = chain.quotes.copy()
    dte = np.array([(e - chain.asof).days for e in q["expiry"]], dtype="float64")
    q["T"] = dte / cfg.day_count
    q = q[(dte >= cfg.expiries.min_days_to_expiry) & (dte <= cfg.expiries.max_days_to_expiry)]
    q = q[(q["bid"] > 0) & (q["ask"] >= q["bid"])]
    q["mid"] = 0.5 * (q["bid"] + q["ask"])
    rel = (q["ask"] - q["bid"]) / q["mid"]
    q = q[rel <= _PARITY_MAX_REL_SPREAD]
    mny = q["strike"] / chain.spot
    q = q[(mny > _PARITY_MONEYNESS[0]) & (mny < _PARITY_MONEYNESS[1])]
    return q[["expiry", "type", "strike", "mid", "T", "spot"]].reset_index(drop=True)


def extract_forwards(parity_df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Fit forward & discount per expiry via the put-call-parity regression.

    ``parity_df`` should be the output of :func:`parity_frame` (or any frame with columns
    ``expiry, type, strike, mid, T, spot``).
    """
    cfg = cfg or Config.load()
    out = []
    for expiry, slice_df in parity_df.groupby("expiry"):
        T = float(slice_df["T"].iloc[0])
        spot = float(slice_df["spot"].iloc[0])
        out.append(_fit_one(expiry, T, spot, slice_df, cfg))
    return pd.DataFrame(out, columns=FORWARD_COLUMNS).sort_values("T").reset_index(drop=True)


def _fit_one(expiry, T: float, spot: float, slice_df: pd.DataFrame, cfg: Config) -> dict:
    # Pair calls and puts at common strikes on their mid prices.
    calls = slice_df[slice_df["type"] == "C"][["strike", "mid"]].rename(columns={"mid": "C"})
    puts = slice_df[slice_df["type"] == "P"][["strike", "mid"]].rename(columns={"mid": "P"})
    pairs = calls.merge(puts, on="strike", how="inner").sort_values("strike")

    fitted = _regress_in_band(pairs, spot, T)
    if fitted is not None:
        disc, F, rate, n_pairs = fitted
        return _row(expiry, T, F, disc, rate, n_pairs, "parity")

    # Fallback: assume cost-of-carry at the configured risk-free rate, no dividends.
    disc = math.exp(-cfg.risk_free * T)
    F = spot / disc
    return _row(expiry, T, F, disc, cfg.risk_free, len(pairs), "fallback")


def _regress_in_band(pairs: pd.DataFrame, spot: float, T: float):
    """OLS of (C-P) on K within the tightest ATM band that has enough pairs.

    Returns ``(disc, F, rate, n_pairs)`` or ``None`` if no sane fit is possible.
    A fit is rejected if the discount factor leaves (0, 1] or the implied rate is outside
    :data:`_RATE_BOUNDS` (symptoms of stale/illiquid quotes).
    """
    if len(pairs) < _MIN_PAIRS:
        return None
    y_all = (pairs["C"] - pairs["P"]).to_numpy()
    k_all = pairs["strike"].to_numpy()
    rel = np.abs(k_all / spot - 1.0)

    for band in _ATM_BANDS:
        sel = rel <= band
        if sel.sum() < _MIN_PAIRS:
            continue
        slope, intercept = np.polyfit(k_all[sel], y_all[sel], 1)
        disc = -slope
        if disc <= 0 or disc > 1.0 + 1e-6:  # discount factor must lie in (0, 1]
            continue
        F = intercept / disc
        if F <= 0:
            continue
        rate = -math.log(disc) / T if T > 0 else math.nan
        if not (_RATE_BOUNDS[0] <= rate <= _RATE_BOUNDS[1]):
            continue
        return float(disc), float(F), float(rate), int(sel.sum())
    return None


def _row(expiry, T, F, disc, rate, n_pairs, method) -> dict:
    return {
        "expiry": expiry, "T": T, "F": F, "disc": disc,
        "rate": rate, "n_pairs": n_pairs, "method": method,
    }


def attach_forward(cleaned: pd.DataFrame, forwards: pd.DataFrame) -> pd.DataFrame:
    """Merge per-expiry F/disc onto quotes and add log-moneyness ``k = log(K/F)``."""
    merged = cleaned.merge(forwards[["expiry", "F", "disc", "rate"]], on="expiry", how="inner")
    merged["log_moneyness"] = np.log(merged["strike"] / merged["F"])
    return merged.sort_values(["T", "type", "strike"]).reset_index(drop=True)


def extract_and_attach(
    chain: RawChain, cleaned: pd.DataFrame, cfg: Config | None = None
) -> ForwardResult:
    """Convenience: extract forwards from the raw chain and attach to the cleaned frame.

    Forwards are fit on a liquidity-light, both-legs selection (:func:`parity_frame`)
    drawn from the raw chain, then merged onto the OI-filtered ``cleaned`` smile points.
    """
    cfg = cfg or Config.load()
    forwards = extract_forwards(parity_frame(chain, cfg), cfg)
    quotes = attach_forward(cleaned, forwards)
    return ForwardResult(forwards=forwards, quotes=quotes)

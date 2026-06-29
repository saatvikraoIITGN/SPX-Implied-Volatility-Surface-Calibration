"""Black-Scholes (Black-76 forward form) pricer and implied-vol inversion.

Working in the **forward measure** is natural once the market-implied forward ``F`` and
discount ``D`` are known (see :mod:`volsurface.forward`):

    call = D * [ F·Φ(d1) - K·Φ(d2) ],   put = D * [ K·Φ(-d2) - F·Φ(-d1) ],
    d1 = (-k + w/2) / √w,  d2 = d1 - √w,  k = log(K/F),  w = σ²T  (total variance).

Implied vol is recovered by inverting the (undiscounted) price for ``σ`` with Brent's
method, bracketed and bisection-backed. We solve on the **out-of-the-money** leg per
strike (calls for k>0, puts for k<0): OTM options carry the vol information with the most
vega and the least intrinsic value, and they are exactly the legs that survive liquidity
filtering. Total variance ``w = σ²T`` is added for the arbitrage and SVI stages.

Jaeckel's "Let's Be Rational" is the production-grade fast inversion; Brent is more than
adequate here and keeps the dependency surface minimal.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

from .config import Config

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def bs_forward_price(F: float, K: float, T: float, sigma: float, opt_type: str) -> float:
    """Undiscounted Black-76 price (i.e. price / discount factor)."""
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        # Degenerate: return intrinsic in forward terms.
        return max(F - K, 0.0) if opt_type == "C" else max(K - F, 0.0)
    sqrt_w = sigma * math.sqrt(T)
    k = math.log(K / F)
    d1 = (-k + 0.5 * sqrt_w * sqrt_w) / sqrt_w
    d2 = d1 - sqrt_w
    if opt_type == "C":
        return F * ndtr(d1) - K * ndtr(d2)
    return K * ndtr(-d2) - F * ndtr(-d1)


def bs_price(F: float, K: float, T: float, sigma: float, disc: float, opt_type: str) -> float:
    """Discounted Black-76 option price."""
    return disc * bs_forward_price(F, K, T, sigma, opt_type)


def bs_vega(F: float, K: float, T: float, sigma: float, disc: float) -> float:
    """Black-76 vega dPrice/dσ (identical for calls and puts)."""
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    sqrt_w = sigma * sqrt_T
    k = math.log(K / F)
    d1 = (-k + 0.5 * sqrt_w * sqrt_w) / sqrt_w
    return disc * F * _norm_pdf(d1) * sqrt_T


def implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    disc: float,
    opt_type: str,
    cfg: Config | None = None,
) -> float:
    """Invert a discounted option price for implied volatility.

    Returns ``nan`` when the price violates the no-arbitrage bounds (so no BS vol exists)
    or the root solve fails to converge.
    """
    cfg = cfg or Config.load()
    s = cfg.iv_solver
    if not (T > 0 and F > 0 and K > 0 and disc > 0 and price > 0):
        return math.nan

    target = price / disc  # undiscounted
    intrinsic = max(F - K, 0.0) if opt_type == "C" else max(K - F, 0.0)
    upper_bound = F if opt_type == "C" else K
    # Price must sit strictly inside (intrinsic, forward/strike) for a finite vol to exist.
    if target <= intrinsic + 1e-12 or target >= upper_bound - 1e-12:
        return math.nan

    def objective(sigma: float) -> float:
        return bs_forward_price(F, K, T, sigma, opt_type) - target

    lo, hi = s.vol_lower, s.vol_upper
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:  # not bracketed within [vol_lower, vol_upper]
        return math.nan
    try:
        return float(brentq(objective, lo, hi, xtol=s.price_tol, maxiter=100))
    except (RuntimeError, ValueError):
        return _bisection(objective, lo, hi, s.price_tol)


def _bisection(objective, lo: float, hi: float, tol: float, max_iter: int = 200) -> float:
    """Bisection fallback if Brent fails to converge."""
    f_lo = objective(lo)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = objective(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def select_otm(quotes: pd.DataFrame, keep_itm: bool = False) -> pd.DataFrame:
    """Keep one out-of-the-money quote per (expiry, strike): call if K>=F, else put.

    In-the-money options carry little vol information and large intrinsic value, so their
    inverted IVs are noisy and unreliable. By default a strike is **dropped** when its OTM
    leg is absent (e.g. filtered out for liquidity) rather than falling back to the ITM
    leg; set ``keep_itm=True`` to fall back instead.

    Expects columns ``expiry, type, strike, F`` (plus the rest carried through).
    """
    preferred = np.where(quotes["strike"] >= quotes["F"], "C", "P")
    quotes = quotes.assign(_pref=preferred)
    keep = []
    for (_, _), grp in quotes.groupby(["expiry", "strike"]):
        pref = grp["_pref"].iloc[0]
        chosen = grp[grp["type"] == pref]
        if len(chosen):
            keep.append(chosen.iloc[0])
        elif keep_itm:
            keep.append(grp.iloc[0])
    out = pd.DataFrame(keep).drop(columns="_pref")
    return out.sort_values(["T", "strike"]).reset_index(drop=True)


def solve_iv_frame(quotes: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Add implied vol ``iv`` and total variance ``w = iv²·T`` to a quote frame.

    Rows whose price admits no BS vol get ``iv = nan`` and are dropped from the result.
    Expects columns ``F, disc, strike, T, mid, type`` (output of forward attachment).
    """
    cfg = cfg or Config.load()
    df = quotes.copy()
    ivs = [
        implied_vol(row.mid, row.F, row.strike, row.T, row.disc, row.type, cfg)
        for row in df.itertuples(index=False)
    ]
    df["iv"] = ivs
    df = df[df["iv"].notna() & (df["iv"] > 0)].copy()
    df["w"] = df["iv"] ** 2 * df["T"]
    return df.sort_values(["T", "strike"]).reset_index(drop=True)

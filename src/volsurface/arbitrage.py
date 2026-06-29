"""No-arbitrage diagnostics for a raw implied-vol grid.

Two static-arbitrage conditions must hold on any tradeable surface:

* **Calendar spread**: total variance ``w(k, T) = σ²(k, T)·T`` must be non-decreasing in
  ``T`` at fixed log-moneyness ``k``. A decrease implies a negative forward variance —
  a calendar-spread arbitrage.
* **Butterfly**: the undiscounted call price must be convex in strike
  (``∂²C/∂K² ≥ 0``), equivalently the risk-neutral density is non-negative. A concave
  kink is a butterfly arbitrage.

This stage *flags and counts* violations on the raw inverted grid (it does not repair
them). Quantifying how many raw points violate — and, later, how the SVI fit removes them
— is the project's "surface diagnostics" outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config

# Tolerances absorb quote noise so we don't flag every rounding wiggle.
_BUTTERFLY_TOL = 1e-6      # on the second divided difference of undiscounted call price
_CALENDAR_TOL = 1e-6      # on total-variance decreases (in variance units)
_CALENDAR_GRID_N = 50     # common k-grid resolution for the calendar check


@dataclass
class ArbitrageReport:
    n_points: int = 0
    n_butterfly: int = 0
    n_calendar: int = 0
    calendar_by_expiry: dict = field(default_factory=dict)

    @property
    def n_violations(self) -> int:
        return self.n_butterfly + self.n_calendar

    def __str__(self) -> str:
        return (
            f"ArbitrageReport: {self.n_points} points | "
            f"butterfly violations={self.n_butterfly} | "
            f"calendar violations={self.n_calendar}"
        )


def undiscounted_call(df: pd.DataFrame) -> np.ndarray:
    """Undiscounted (forward) call price for each quote, via parity for puts.

    ``C_fwd = mid/disc`` for calls; ``C_fwd = P_fwd + (F - K)`` for puts.
    """
    fwd_price = df["mid"].to_numpy() / df["disc"].to_numpy()
    is_put = (df["type"] == "P").to_numpy()
    fwd_price = fwd_price + is_put * (df["F"].to_numpy() - df["strike"].to_numpy())
    return fwd_price


def flag_butterfly(iv_frame: pd.DataFrame) -> pd.DataFrame:
    """Annotate each point with ``butterfly_ok`` (call price convex in strike).

    For three consecutive strikes the second divided difference of the undiscounted call
    price must be ``>= -tol``; the middle strike is flagged when it is not.
    """
    df = iv_frame.copy()
    df["_cfwd"] = undiscounted_call(df)
    df["butterfly_ok"] = True
    for _, idx in df.groupby("expiry").groups.items():
        sub = df.loc[idx].sort_values("strike")
        K = sub["strike"].to_numpy()
        C = sub["_cfwd"].to_numpy()
        rows = sub.index.to_numpy()
        for i in range(1, len(K) - 1):
            h1, h2 = K[i] - K[i - 1], K[i + 1] - K[i]
            if h1 <= 0 or h2 <= 0:
                continue
            second = (C[i + 1] - C[i]) / h2 - (C[i] - C[i - 1]) / h1
            second /= 0.5 * (K[i + 1] - K[i - 1])
            if second < -_BUTTERFLY_TOL:
                df.loc[rows[i], "butterfly_ok"] = False
    return df.drop(columns="_cfwd")


def flag_calendar(iv_frame: pd.DataFrame) -> tuple[dict, int]:
    """Count calendar violations: total variance decreasing in T at fixed k.

    Each expiry's ``w(k)`` is linearly interpolated onto a common k-grid (over the range it
    actually covers); at each grid node we check monotonicity across sorted maturities.
    Returns ``(violations_per_expiry, total)``.
    """
    slices = []
    for expiry, sub in iv_frame.groupby("expiry"):
        s = sub.sort_values("log_moneyness")
        T = float(s["T"].iloc[0])
        slices.append((T, expiry, s["log_moneyness"].to_numpy(), s["w"].to_numpy()))
    slices.sort(key=lambda x: x[0])
    if len(slices) < 2:
        return {}, 0

    kmin = max(s[2].min() for s in slices)
    kmax = min(s[2].max() for s in slices)
    per_expiry: dict = {}
    total = 0
    if kmax <= kmin:
        return per_expiry, 0  # no overlapping moneyness range across expiries
    grid = np.linspace(kmin, kmax, _CALENDAR_GRID_N)

    prev_w = None
    for _T, expiry, k, w in slices:
        w_grid = np.interp(grid, k, w)
        if prev_w is not None:
            viol = int(np.sum(w_grid < prev_w - _CALENDAR_TOL))
            if viol:
                per_expiry[expiry] = viol
                total += viol
        prev_w = w_grid
    return per_expiry, total


def check_arbitrage(iv_frame: pd.DataFrame, cfg: Config | None = None) -> tuple[pd.DataFrame, ArbitrageReport]:
    """Run both checks; return the butterfly-annotated frame and a report."""
    flagged = flag_butterfly(iv_frame)
    cal_by_expiry, n_cal = flag_calendar(iv_frame)
    report = ArbitrageReport(
        n_points=len(iv_frame),
        n_butterfly=int((~flagged["butterfly_ok"]).sum()),
        n_calendar=n_cal,
        calendar_by_expiry=cal_by_expiry,
    )
    return flagged, report


def drop_butterfly_violations(flagged: pd.DataFrame) -> pd.DataFrame:
    """Remove points flagged as butterfly-violating."""
    return flagged[flagged["butterfly_ok"]].drop(columns="butterfly_ok").reset_index(drop=True)

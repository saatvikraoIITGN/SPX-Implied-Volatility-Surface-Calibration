"""Clean and filter a raw option chain.

Raw chains are full of unusable quotes: zero/one-sided markets, crossed quotes, illiquid
strikes, stale wings. This stage drops them, computes mid prices and moneyness, restricts
to a sane moneyness/expiry window, and returns both the cleaned frame and a
:class:`RejectionReport` attributing every dropped quote to the first rule it failed.

The report is the seed of the project's "surface diagnostics" outcome (how many raw points
get rejected, and why).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .sources.base import RawChain

# Columns the cleaned frame carries forward (raw columns + derived).
DERIVED_COLUMNS = ["T", "mid", "spread", "rel_spread", "moneyness"]
CLEANED_COLUMNS = [
    "expiry", "type", "strike", "bid", "ask", "last",
    "volume", "open_interest", "spot",
] + DERIVED_COLUMNS


@dataclass
class RejectionReport:
    """Per-rule counts of dropped quotes (sequential attribution)."""

    n_input: int = 0
    n_output: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    expiries_dropped: int = 0

    def add(self, reason: str, count: int) -> None:
        if count:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(count)

    @property
    def n_dropped(self) -> int:
        return self.n_input - self.n_output

    def to_frame(self) -> pd.DataFrame:
        rows = [{"reason": r, "count": c} for r, c in self.dropped.items()]
        rows.append({"reason": "SURVIVED", "count": self.n_output})
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        lines = [f"RejectionReport: {self.n_input} in -> {self.n_output} out "
                 f"({self.n_dropped} dropped, {self.expiries_dropped} expiries dropped)"]
        for reason, count in self.dropped.items():
            lines.append(f"  - {reason}: {count}")
        return "\n".join(lines)


def clean_chain(chain: RawChain, cfg: Config | None = None) -> tuple[pd.DataFrame, RejectionReport]:
    """Filter a raw chain into a clean quote frame.

    Returns
    -------
    (cleaned, report):
        ``cleaned`` has columns :data:`CLEANED_COLUMNS`, one row per surviving quote.
        ``report`` attributes every dropped quote to a rule.
    """
    cfg = cfg or Config.load()
    cc = cfg.cleaning
    df = chain.quotes.copy()
    report = RejectionReport(n_input=len(df))

    # Time to expiry (years). Drop non-positive and out-of-window tenors.
    dte = np.array([(e - chain.asof).days for e in df["expiry"]], dtype="float64")
    df["T"] = dte / cfg.day_count
    mask = dte > 0
    report.add("non_positive_dte", (~mask).sum())
    df, mask_full = _apply(df, mask)

    win = (df["T"] * cfg.day_count >= cfg.expiries.min_days_to_expiry) & (
        df["T"] * cfg.day_count <= cfg.expiries.max_days_to_expiry
    )
    report.add("dte_out_of_window", (~win).sum())
    df, _ = _apply(df, win)

    # Quote-quality filters (sequential: a row is attributed to the first rule it fails).
    if cc.require_positive_bid:
        m = df["bid"] > 0
        report.add("non_positive_bid", (~m).sum())
        df, _ = _apply(df, m)

    m = df["ask"] > 0
    report.add("non_positive_ask", (~m).sum())
    df, _ = _apply(df, m)

    if cc.drop_crossed:
        m = df["ask"] >= df["bid"]
        report.add("crossed_market", (~m).sum())
        df, _ = _apply(df, m)

    m = df["open_interest"] >= cc.min_open_interest
    report.add("low_open_interest", (~m).sum())
    df, _ = _apply(df, m)

    if cc.min_volume > 0:
        m = df["volume"] >= cc.min_volume
        report.add("low_volume", (~m).sum())
        df, _ = _apply(df, m)

    # Derived prices.
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["spread"] = df["ask"] - df["bid"]
    df["rel_spread"] = df["spread"] / df["mid"].replace(0.0, np.nan)

    m = df["rel_spread"] <= cc.max_rel_spread
    report.add("wide_spread", (~m).sum())
    df, _ = _apply(df, m)

    # Moneyness window K/S.
    df["moneyness"] = df["strike"] / chain.spot
    m = (df["moneyness"] > cc.moneyness_min) & (df["moneyness"] < cc.moneyness_max)
    report.add("moneyness_out_of_range", (~m).sum())
    df, _ = _apply(df, m)

    # Drop thin expiry slices (need enough strikes to fit a smile).
    strikes_per_expiry = df.groupby("expiry")["strike"].nunique()
    keep_exp = strikes_per_expiry[strikes_per_expiry >= cc.min_strikes_per_expiry].index
    thin = ~df["expiry"].isin(keep_exp)
    report.expiries_dropped = int((~strikes_per_expiry.index.isin(keep_exp)).sum())
    report.add("thin_expiry_slice", thin.sum())
    df, _ = _apply(df, ~thin)

    cleaned = df.loc[:, CLEANED_COLUMNS].sort_values(["expiry", "type", "strike"]).reset_index(drop=True)
    report.n_output = len(cleaned)
    return cleaned, report


def _apply(df: pd.DataFrame, mask: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Keep rows where ``mask`` is True, returning the filtered frame."""
    kept = df.loc[mask].copy()
    return kept, mask

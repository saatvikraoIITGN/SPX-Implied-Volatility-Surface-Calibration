"""Assemble the fitted SVI slices into a surface and render diagnostics plots.

Three views, all standard practitioner diagnostics:
  1. 3D implied-vol surface over (log-moneyness, maturity).
  2. Per-expiry smile: market IV scatter + bid/ask IV band + fitted SVI curve.
  3. ATM term structure: ATM (k=0) implied vol vs maturity.

Plots are written to the configured output directory; a non-interactive backend is used so
this runs headless.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402
from .iv_solver import implied_vol  # noqa: E402
from .svi import params_from_row  # noqa: E402


@dataclass
class SurfaceGrid:
    k: np.ndarray              # log-moneyness grid (1D)
    T: np.ndarray              # maturities (1D), ascending
    iv: np.ndarray             # implied vol matrix, shape (len(T), len(k))


def evaluate_surface(svi_df: pd.DataFrame, k_grid: np.ndarray | None = None) -> SurfaceGrid:
    """Evaluate fitted SVI implied vol on a (maturity x log-moneyness) grid."""
    if k_grid is None:
        k_grid = np.linspace(-0.30, 0.15, 60)
    svi_df = svi_df.sort_values("T")
    Ts = svi_df["T"].to_numpy()
    iv = np.empty((len(svi_df), len(k_grid)))
    for i, (_, row) in enumerate(svi_df.iterrows()):
        iv[i, :] = params_from_row(row).implied_vol(k_grid, row["T"])
    return SurfaceGrid(k=np.asarray(k_grid), T=Ts, iv=iv)


def atm_term_structure(svi_df: pd.DataFrame) -> pd.DataFrame:
    """ATM (k=0, i.e. K=F) implied vol per maturity."""
    rows = []
    for _, row in svi_df.sort_values("T").iterrows():
        p = params_from_row(row)
        w0 = float(np.maximum(p.total_variance(0.0), 0.0))
        rows.append({"T": row["T"], "atm_iv": (w0 / row["T"]) ** 0.5})
    return pd.DataFrame(rows)


def compute_iv_band(slice_points: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Add iv_bid / iv_ask columns by inverting the bid and ask prices of each quote."""
    cfg = cfg or Config.load()
    df = slice_points.copy()
    df["iv_bid"] = [
        implied_vol(r.bid, r.F, r.strike, r.T, r.disc, r.type, cfg)
        for r in df.itertuples(index=False)
    ]
    df["iv_ask"] = [
        implied_vol(r.ask, r.F, r.strike, r.T, r.disc, r.type, cfg)
        for r in df.itertuples(index=False)
    ]
    return df


def _out(cfg: Config, name: str) -> Path:
    d = cfg.paths.resolve("output_dir")
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def plot_surface_3d(svi_df: pd.DataFrame, cfg: Config | None = None, name: str = "surface_3d.png") -> Path:
    cfg = cfg or Config.load()
    grid = evaluate_surface(svi_df)
    K, T = np.meshgrid(grid.k, grid.T)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K, T, grid.iv * 100, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("log-moneyness  k = log(K/F)")
    ax.set_ylabel("maturity T (yrs)")
    ax.set_zlabel("implied vol (%)")
    ax.set_title("SPX SVI implied-volatility surface")
    path = _out(cfg, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_smile(
    svi_df: pd.DataFrame,
    iv_points: pd.DataFrame,
    expiry,
    cfg: Config | None = None,
    name: str | None = None,
) -> Path:
    """Smile for one expiry: market IV + bid/ask band + fitted SVI curve."""
    cfg = cfg or Config.load()
    row = svi_df[svi_df["expiry"] == expiry].iloc[0]
    p = params_from_row(row)
    pts = compute_iv_band(iv_points[iv_points["expiry"] == expiry].sort_values("log_moneyness"), cfg)
    k = pts["log_moneyness"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    band_ok = pts["iv_bid"].notna() & pts["iv_ask"].notna()
    ax.fill_between(
        k[band_ok], pts["iv_bid"][band_ok] * 100, pts["iv_ask"][band_ok] * 100,
        color="gray", alpha=0.3, label="bid/ask IV band",
    )
    ax.scatter(k, pts["iv"] * 100, s=14, color="steelblue", label="market mid IV", zorder=3)
    kk = np.linspace(k.min(), k.max(), 200)
    ax.plot(kk, p.implied_vol(kk, row["T"]) * 100, color="crimson", lw=2, label="SVI fit")
    ax.set_xlabel("log-moneyness  k = log(K/F)")
    ax.set_ylabel("implied vol (%)")
    ax.set_title(f"SPX smile  {expiry}  (T={row['T']:.3f}y, RMSE={row['rmse_vol']*100:.2f} vol pts)")
    ax.legend()
    path = _out(cfg, name or f"smile_{expiry}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_atm_term_structure(svi_df: pd.DataFrame, cfg: Config | None = None, name: str = "atm_term_structure.png") -> Path:
    cfg = cfg or Config.load()
    ts = atm_term_structure(svi_df)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts["T"], ts["atm_iv"] * 100, "o-", color="darkgreen")
    ax.set_xlabel("maturity T (yrs)")
    ax.set_ylabel("ATM implied vol (%)")
    ax.set_title("SPX ATM term structure")
    ax.grid(alpha=0.3)
    path = _out(cfg, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path

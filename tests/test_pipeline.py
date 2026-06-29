"""End-to-end pipeline test on a synthetic SVI-consistent chain (no network)."""
from datetime import date

import numpy as np
import pandas as pd

from volsurface.config import Config
from volsurface.iv_solver import bs_price
from volsurface.pipeline import run_pipeline
from volsurface.sources.base import ChainSource, RawChain, normalize_quotes
from volsurface.svi import SVIParams

ASOF = date(2026, 6, 29)
SPOT = 5000.0


class SyntheticSource(ChainSource):
    """A clean, arbitrage-free chain generated from known SVI slices."""

    def fetch(self, asof=None):
        rows = []
        for days, params in [
            (30, SVIParams(0.010, 0.08, -0.5, -0.02, 0.10)),
            (90, SVIParams(0.020, 0.10, -0.5, -0.02, 0.12)),
            (180, SVIParams(0.035, 0.12, -0.45, -0.02, 0.14)),
        ]:
            T = days / 365.25
            F = SPOT * np.exp(0.04 * T)
            disc = np.exp(-0.04 * T)
            for k in np.linspace(-0.20, 0.12, 25):
                K = F * np.exp(k)
                iv = float(params.implied_vol(k, T))
                for typ in ("C", "P"):
                    price = bs_price(F, K, T, iv, disc, typ)
                    spread = max(0.05, 0.01 * price)
                    rows.append({
                        "expiry": asof.fromordinal(asof.toordinal() + days) if asof
                        else date.fromordinal(ASOF.toordinal() + days),
                        "type": typ, "strike": float(K),
                        "bid": price - spread, "ask": price + spread, "last": price,
                        "volume": 50, "open_interest": 500, "spot": SPOT,
                    })
        quotes = normalize_quotes(pd.DataFrame(rows))
        return RawChain(quotes=quotes, asof=asof or ASOF, spot=SPOT, ticker="^SPX")


def test_pipeline_end_to_end(tmp_path):
    cfg = Config.load()
    # Redirect all output dirs into the temp area.
    object.__setattr__(cfg.paths, "raw_dir", str(tmp_path / "raw"))
    object.__setattr__(cfg.paths, "processed_dir", str(tmp_path / "processed"))
    object.__setattr__(cfg.paths, "output_dir", str(tmp_path / "outputs"))

    result = run_pipeline(cfg, source=SyntheticSource(), make_plots=True)

    d = result.diagnostics
    assert d.n_raw_quotes > 0
    assert d.n_iv_points > 0
    assert d.n_slices_fit == 3
    assert d.all_slices_butterfly_free
    assert d.median_rmse_vol < 0.01          # sub-1-vol-point fit on clean data
    # Clean synthetic data should have (near) zero raw arbitrage violations.
    assert d.butterfly_violations_raw <= 2

    # Plots and processed artifacts were written.
    assert all(__import__("pathlib").Path(p).exists() for p in result.plots.values())
    assert (tmp_path / "processed").exists()


def test_pipeline_persists_svi_params(tmp_path):
    cfg = Config.load()
    object.__setattr__(cfg.paths, "raw_dir", str(tmp_path / "raw"))
    object.__setattr__(cfg.paths, "processed_dir", str(tmp_path / "processed"))
    object.__setattr__(cfg.paths, "output_dir", str(tmp_path / "outputs"))
    run_pipeline(cfg, source=SyntheticSource(), make_plots=False)
    parquets = list((tmp_path / "processed").glob("svi_params_*.parquet"))
    assert len(parquets) == 1
    saved = pd.read_parquet(parquets[0])
    assert len(saved) == 3
    assert "butterfly_free" in saved.columns

"""End-to-end pipeline: raw SPX chain -> arbitrage-free SVI surface + analytics.

Wires every stage together and produces the project's **surface-diagnostics** summary:
how many raw quotes are rejected, how many butterfly/calendar violations the raw inverted
grid contains, and that the fitted SVI surface is butterfly-free. Run as a module:

    python -m volsurface.pipeline --asof today          # fetch a fresh snapshot
    python -m volsurface.pipeline --use-cached          # reuse the latest saved snapshot
"""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import asdict, dataclass, field
from datetime import date

import pandas as pd

from .arbitrage import ArbitrageReport, check_arbitrage, drop_butterfly_violations
from .cleaner import RejectionReport, clean_chain
from .config import Config
from .density import plot_rnd
from .fetcher import fetch_chain, load_raw
from .forward import extract_and_attach
from .iv_solver import select_otm, solve_iv_frame
from .sources.base import ChainSource, RawChain
from .surface import plot_atm_term_structure, plot_smile, plot_surface_3d
from .svi import fit_surface


@dataclass
class Diagnostics:
    asof: str = ""
    spot: float = 0.0
    n_raw_quotes: int = 0
    n_after_clean: int = 0
    n_iv_points: int = 0
    n_expiries: int = 0
    butterfly_violations_raw: int = 0
    butterfly_pct_raw: float = 0.0
    calendar_violations_raw: int = 0
    n_slices_fit: int = 0
    all_slices_butterfly_free: bool = False
    median_rmse_vol: float = 0.0
    rejection_breakdown: dict = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            "===== SPX volatility-surface diagnostics =====",
            f"as-of {self.asof}   spot {self.spot:.2f}",
            f"raw quotes ............... {self.n_raw_quotes}",
            f"after cleaning ........... {self.n_after_clean}",
            f"OTM points with IV ....... {self.n_iv_points}  across {self.n_expiries} expiries",
            f"raw butterfly violations . {self.butterfly_violations_raw} "
            f"({self.butterfly_pct_raw:.1f}% of points)",
            f"raw calendar violations .. {self.calendar_violations_raw}",
            f"SVI slices fit ........... {self.n_slices_fit}  "
            f"(all butterfly-free: {self.all_slices_butterfly_free})",
            f"median fit RMSE .......... {self.median_rmse_vol*100:.2f} vol points",
            "==============================================",
        ]
        return "\n".join(lines)


@dataclass
class PipelineResult:
    chain: RawChain
    cleaned: pd.DataFrame
    rejection: RejectionReport
    forwards: pd.DataFrame
    iv_frame: pd.DataFrame          # arbitrage-annotated, butterfly violations dropped
    arb: ArbitrageReport
    svi_params: pd.DataFrame
    diagnostics: Diagnostics
    plots: dict = field(default_factory=dict)


def run_pipeline(
    cfg: Config | None = None,
    asof: date | None = None,
    source: ChainSource | None = None,
    use_cached: bool = False,
    make_plots: bool = True,
    persist: bool = True,
) -> PipelineResult:
    cfg = cfg or Config.load()

    # 1. Ingest -----------------------------------------------------------------
    chain = _load_cached(cfg) if use_cached else fetch_chain(cfg, source=source, asof=asof)

    # 2. Clean ------------------------------------------------------------------
    cleaned, rejection = clean_chain(chain, cfg)

    # 3. Forward & log-moneyness ------------------------------------------------
    fwd = extract_and_attach(chain, cleaned, cfg)

    # 4. OTM selection + IV solve ----------------------------------------------
    iv_all = solve_iv_frame(select_otm(fwd.quotes), cfg)

    # 5. No-arbitrage diagnostics (on the raw inverted grid) --------------------
    flagged, arb = check_arbitrage(iv_all, cfg)
    iv_clean = drop_butterfly_violations(flagged)

    # 6. SVI calibration --------------------------------------------------------
    svi_params = fit_surface(iv_clean, cfg)

    # 7. Diagnostics ------------------------------------------------------------
    diag = _diagnostics(chain, rejection, iv_all, arb, svi_params)

    plots: dict = {}
    if make_plots and len(svi_params):
        plots["surface"] = str(plot_surface_3d(svi_params, cfg))
        plots["atm_term_structure"] = str(plot_atm_term_structure(svi_params, cfg))
        plots["rnd"] = str(plot_rnd(svi_params, cfg=cfg))
        mid_expiry = svi_params.sort_values("T").iloc[len(svi_params) // 2]["expiry"]
        plots["smile"] = str(plot_smile(svi_params, iv_clean, mid_expiry, cfg, name="smile.png"))

    if persist:
        _persist(cfg, chain.asof, svi_params, diag)

    return PipelineResult(
        chain=chain, cleaned=cleaned, rejection=rejection, forwards=fwd.forwards,
        iv_frame=iv_clean, arb=arb, svi_params=svi_params, diagnostics=diag, plots=plots,
    )


def _load_cached(cfg: Config) -> RawChain:
    raw_dir = cfg.paths.resolve("raw_dir")
    files = sorted(glob.glob(str(raw_dir / "spx_*.parquet")))
    if not files:
        raise FileNotFoundError(f"no cached snapshots in {raw_dir}; run without --use-cached")
    return load_raw(files[-1])


def _diagnostics(chain, rejection, iv_all, arb, svi_params) -> Diagnostics:
    n_pts = len(iv_all)
    return Diagnostics(
        asof=chain.asof.isoformat(),
        spot=float(chain.spot),
        n_raw_quotes=rejection.n_input,
        n_after_clean=rejection.n_output,
        n_iv_points=n_pts,
        n_expiries=int(iv_all["expiry"].nunique()) if n_pts else 0,
        butterfly_violations_raw=arb.n_butterfly,
        butterfly_pct_raw=100.0 * arb.n_butterfly / n_pts if n_pts else 0.0,
        calendar_violations_raw=arb.n_calendar,
        n_slices_fit=len(svi_params),
        all_slices_butterfly_free=bool(svi_params["butterfly_free"].all()) if len(svi_params) else False,
        median_rmse_vol=float(svi_params["rmse_vol"].median()) if len(svi_params) else 0.0,
        rejection_breakdown=dict(rejection.dropped),
    )


def _persist(cfg: Config, asof: date, svi_params: pd.DataFrame, diag: Diagnostics) -> None:
    out = cfg.paths.resolve("processed_dir")
    out.mkdir(parents=True, exist_ok=True)
    if len(svi_params):
        svi_params.to_parquet(out / f"svi_params_{asof.isoformat()}.parquet", index=False)
    with open(out / f"diagnostics_{asof.isoformat()}.json", "w") as fh:
        json.dump(asdict(diag), fh, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an arbitrage-free SPX vol surface.")
    parser.add_argument("--asof", default="today", help="snapshot date YYYY-MM-DD or 'today'")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--use-cached", action="store_true", help="reuse latest saved snapshot")
    parser.add_argument("--no-plots", action="store_true", help="skip plot generation")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    asof = None if args.asof == "today" else date.fromisoformat(args.asof)
    result = run_pipeline(cfg, asof=asof, use_cached=args.use_cached, make_plots=not args.no_plots)
    print(result.diagnostics.render())
    if result.plots:
        print("\nplots written:")
        for name, path in result.plots.items():
            print(f"  {name:18s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

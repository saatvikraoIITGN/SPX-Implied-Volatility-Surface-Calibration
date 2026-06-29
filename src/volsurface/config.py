"""Typed configuration loaded from a YAML file.

All pipeline tunables (rates, cleaning thresholds, solver tolerances, SVI bounds, paths)
are centralized here so that no stage hard-codes magic numbers. Load once with
``Config.load()`` and pass the resulting object down the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Repository root = three parents up from this file: src/volsurface/config.py -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class UnderlyingConfig:
    ticker: str = "^SPX"
    fallback_tickers: list[str] = field(default_factory=lambda: ["^SPXW", "SPY"])


@dataclass(frozen=True)
class CleaningConfig:
    min_open_interest: int = 10
    min_volume: int = 0
    require_positive_bid: bool = True
    drop_crossed: bool = True
    moneyness_min: float = 0.7
    moneyness_max: float = 1.3
    max_rel_spread: float = 1.5
    min_strikes_per_expiry: int = 6


@dataclass(frozen=True)
class ExpiryConfig:
    min_days_to_expiry: int = 7
    max_days_to_expiry: int = 365


@dataclass(frozen=True)
class IVSolverConfig:
    vol_lower: float = 1.0e-4
    vol_upper: float = 5.0
    price_tol: float = 1.0e-8


@dataclass(frozen=True)
class SVIConfig:
    rho_abs_max: float = 0.999
    sigma_min: float = 1.0e-4
    b_min: float = 0.0
    n_multistart: int = 8


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    output_dir: str = "outputs"

    def resolve(self, name: str) -> Path:
        """Resolve a configured directory to an absolute path under the repo root."""
        rel = getattr(self, name)
        p = (REPO_ROOT / rel).resolve()
        return p


@dataclass(frozen=True)
class Config:
    underlying: UnderlyingConfig = field(default_factory=UnderlyingConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    expiries: ExpiryConfig = field(default_factory=ExpiryConfig)
    iv_solver: IVSolverConfig = field(default_factory=IVSolverConfig)
    svi: SVIConfig = field(default_factory=SVIConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    risk_free: float = 0.04
    day_count: float = 365.25

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load configuration from a YAML file, falling back to dataclass defaults.

        Unknown keys in the YAML are ignored; missing keys fall back to defaults.
        """
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        data = {}
        if path.exists():
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}

        rates = data.get("rates", {}) or {}
        return cls(
            underlying=UnderlyingConfig(**_filter(UnderlyingConfig, data.get("underlying", {}))),
            cleaning=CleaningConfig(**_filter(CleaningConfig, data.get("cleaning", {}))),
            expiries=ExpiryConfig(**_filter(ExpiryConfig, data.get("expiries", {}))),
            iv_solver=IVSolverConfig(**_filter(IVSolverConfig, data.get("iv_solver", {}))),
            svi=SVIConfig(**_filter(SVIConfig, data.get("svi", {}))),
            paths=PathsConfig(**_filter(PathsConfig, data.get("paths", {}))),
            risk_free=float(rates.get("risk_free", 0.04)),
            day_count=float(data.get("day_count", 365.25)),
        )


def _filter(klass, mapping: dict) -> dict:
    """Keep only keys that are valid fields of the given dataclass."""
    if not mapping:
        return {}
    valid = set(getattr(klass, "__dataclass_fields__", {}).keys())
    return {k: v for k, v in mapping.items() if k in valid}

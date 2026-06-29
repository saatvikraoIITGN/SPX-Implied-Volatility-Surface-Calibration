"""Tests for configuration loading."""
from pathlib import Path

from volsurface.config import DEFAULT_CONFIG_PATH, Config


def test_defaults_load_without_file():
    """A missing config path should fall back to dataclass defaults."""
    cfg = Config.load(Path("/nonexistent/path/config.yaml"))
    assert cfg.underlying.ticker == "^SPX"
    assert cfg.cleaning.moneyness_min < cfg.cleaning.moneyness_max
    assert 0 < cfg.risk_free < 1


def test_default_yaml_parses():
    """The shipped default.yaml must parse into a Config."""
    assert DEFAULT_CONFIG_PATH.exists()
    cfg = Config.load(DEFAULT_CONFIG_PATH)
    assert cfg.underlying.ticker == "^SPX"
    assert cfg.cleaning.min_open_interest >= 0
    assert cfg.svi.rho_abs_max < 1.0
    assert cfg.day_count > 0


def test_paths_resolve_to_absolute():
    cfg = Config.load(DEFAULT_CONFIG_PATH)
    raw = cfg.paths.resolve("raw_dir")
    assert raw.is_absolute()
    assert raw.name == "raw"

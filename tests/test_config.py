"""Strict YAML config loading: shipped defaults, rejection of unknown keys,
and proof that YAML values actually change the loaded config."""

from __future__ import annotations

from pathlib import Path

import pytest

from macro_forecasting_lab.config import DEFAULT_CONFIG, config_from_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_YAML = REPO_ROOT / "configs" / "default.yaml"


def test_shipped_yaml_loads_documented_defaults() -> None:
    """configs/default.yaml must reproduce DEFAULT_CONFIG field by field."""
    loaded = config_from_yaml(SHIPPED_YAML)
    for field in DEFAULT_CONFIG.__dataclass_fields__:
        assert getattr(loaded, field) == getattr(
            DEFAULT_CONFIG, field
        ), f"YAML/config mismatch for field {field!r}"


def test_shipped_yaml_has_only_flat_top_level_keys() -> None:
    """The YAML must be flat (no nested `split:`/`experiment:` blocks) and
    every key must name a Config field."""
    import yaml

    payload = yaml.safe_load(SHIPPED_YAML.read_text(encoding="utf-8"))
    allowed = set(DEFAULT_CONFIG.__dataclass_fields__)
    unknown = set(payload) - allowed
    assert not unknown, f"non-Config keys in shipped YAML: {unknown}"
    # Guard against re-introducing the historical nested blocks.
    assert isinstance(payload.get("split", None), type(None))
    assert isinstance(payload.get("experiment", None), type(None))


def test_unknown_key_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("seed: 42\nnot_a_real_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_real_field"):
        config_from_yaml(config_file)


def test_yaml_values_change_loaded_config(tmp_path: Path) -> None:
    """Changing seed/horizons (and other fields) in YAML must be honored —
    the loader cannot silently ignore known keys."""
    config_file = tmp_path / "custom.yaml"
    config_file.write_text(
        "seed: 7\n"
        "horizons: [1, 3]\n"
        "seasonal_period: 4\n"
        "confidence_level: 0.90\n"
        "dm_max_lag: 2\n"
        "target_step: 1\n"
        'train_end: "2005-12-31"\n',
        encoding="utf-8",
    )
    loaded = config_from_yaml(config_file)
    assert loaded.seed == 7
    assert loaded.horizons == (1, 3)
    assert loaded.seasonal_period == 4
    assert loaded.confidence_level == pytest.approx(0.90)
    assert loaded.dm_max_lag == 2
    assert loaded.target_step == 1
    assert loaded.train_end == "2005-12-31"


def test_dm_max_lag_auto_alias_and_validation(tmp_path: Path) -> None:
    auto_file = tmp_path / "auto.yaml"
    auto_file.write_text("dm_max_lag: auto\n", encoding="utf-8")
    assert config_from_yaml(auto_file).dm_max_lag == "auto"

    sentinel_file = tmp_path / "sentinel.yaml"
    sentinel_file.write_text("dm_max_lag: -1\n", encoding="utf-8")
    assert config_from_yaml(sentinel_file).dm_max_lag == "auto"

    bad_string = tmp_path / "bad_string.yaml"
    bad_string.write_text("dm_max_lag: yesterday\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dm_max_lag"):
        config_from_yaml(bad_string)

    bad_int = tmp_path / "bad_int.yaml"
    bad_int.write_text("dm_max_lag: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dm_max_lag"):
        config_from_yaml(bad_int)


def test_invalid_confidence_level_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "bad_cl.yaml"
    config_file.write_text("confidence_level: 1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="confidence_level"):
        config_from_yaml(config_file)

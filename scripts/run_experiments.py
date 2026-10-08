"""CLI entry point: run the full forecasting study and write reports/figures."""

from __future__ import annotations

import argparse

import yaml

from macro_forecasting_lab.config import DEFAULT_CONFIG, Config, SeriesSpec
from macro_forecasting_lab.pipeline import run


def config_from_yaml(path: str) -> Config:
    """Build a Config from configs/default.yaml (falls back to defaults)."""
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    series = DEFAULT_CONFIG.series
    if "series" in payload:
        series = tuple(SeriesSpec(**spec) for spec in payload["series"])
    return Config(
        series=series,
        train_end=payload.get("train_end", DEFAULT_CONFIG.train_end),
        val_start=payload.get("val_start", DEFAULT_CONFIG.val_start),
        val_end=payload.get("val_end", DEFAULT_CONFIG.val_end),
        test_start=payload.get("test_start", DEFAULT_CONFIG.test_start),
        horizons=tuple(payload.get("horizons", DEFAULT_CONFIG.horizons)),
        seed=payload.get("seed", DEFAULT_CONFIG.seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the macro-forecasting experiments end to end."
    )
    parser.add_argument(
        "--config", type=str, default=None, help="path to a YAML config (optional)"
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="force re-download of FRED series",
    )
    args = parser.parse_args()

    config = config_from_yaml(args.config) if args.config else DEFAULT_CONFIG
    results, _artifacts = run(config, refresh=args.refresh_data)

    print("\n=== headline (full test window, best model per series/horizon) ===")
    for (series, horizon), group in results.groupby(["series", "horizon"]):
        full = group[group.window == "full"]
        best = full.loc[full.rmse.idxmin()]
        print(f"{series:22s} h={horizon:<3d} best={best.model:20s} "
              f"RMSE={best.rmse:.3f} MASE={best.mase:.3f}")


if __name__ == "__main__":
    main()

"""CLI entry point: run the full forecasting study and write reports/figures."""

from __future__ import annotations

import argparse

from macro_forecasting_lab.config import DEFAULT_CONFIG, config_from_yaml
from macro_forecasting_lab.pipeline import run

__all__ = ["config_from_yaml", "main"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the macro-forecasting experiments end to end."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to a YAML config (optional; keys must be Config fields, "
        "unknown keys raise)",
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

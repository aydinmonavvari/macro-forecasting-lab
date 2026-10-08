"""CLI entry point: download + cache the FRED series used by the study."""

from __future__ import annotations

import argparse

from macro_forecasting_lab.config import DEFAULT_CONFIG
from macro_forecasting_lab.data import build_modeling_frame, load_series


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download FRED series to data/raw/ and build the modeling frame."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download even if cached CSVs exist",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="download a single FRED series id (e.g. CPIAUCSL) and exit",
    )
    args = parser.parse_args()

    if args.only:
        series = load_series(args.only, raw_dir=DEFAULT_CONFIG.raw_dir, refresh=args.refresh)
        print(f"{args.only}: {len(series)} observations "
              f"({series.index.min().date()} .. {series.index.max().date()})")
        return

    frame = build_modeling_frame(DEFAULT_CONFIG, refresh=args.refresh)
    print(f"modeling frame: {len(frame)} months "
          f"({frame.index.min().date()} .. {frame.index.max().date()})")
    for column in frame.columns:
        series = frame[column].dropna()
        print(f"  {column:28s} n={len(series):4d}  mean={series.mean():9.3f}  "
              f"std={series.std():8.3f}")


if __name__ == "__main__":
    main()

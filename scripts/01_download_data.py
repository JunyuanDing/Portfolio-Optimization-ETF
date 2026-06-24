#!/usr/bin/env python3
"""Download raw adjusted close ETF prices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio.data import DEFAULT_TICKERS, download_adjusted_close, save_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-02-01")
    parser.add_argument("--output", default="data/raw/etf_adjusted_close.csv")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = download_adjusted_close(args.tickers, start=args.start, end=args.end)
    save_frame(prices, PROJECT_ROOT / args.output)
    print(f"Saved raw adjusted close prices to {args.output}. Shape: {prices.shape}")


if __name__ == "__main__":
    main()

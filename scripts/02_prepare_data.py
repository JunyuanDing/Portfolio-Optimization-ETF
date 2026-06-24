#!/usr/bin/env python3
"""Clean raw prices and generate daily returns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio.data import DEFAULT_TICKERS, clean_prices, load_frame, prices_to_returns, save_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/etf_adjusted_close.csv")
    parser.add_argument("--prices-output", default="data/processed/prices.csv")
    parser.add_argument("--returns-output", default="data/processed/returns.csv")
    parser.add_argument("--min-non-missing", type=float, default=0.95)
    parser.add_argument("--end-date", default="2026-01-31")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_prices = load_frame(PROJECT_ROOT / args.input)
    if args.end_date:
        raw_prices = raw_prices.loc[: args.end_date]
    prices = clean_prices(
        raw_prices,
        tickers=DEFAULT_TICKERS,
        min_non_missing=args.min_non_missing,
    )
    returns = prices_to_returns(prices)
    save_frame(prices, PROJECT_ROOT / args.prices_output)
    save_frame(returns, PROJECT_ROOT / args.returns_output)
    print(f"Saved cleaned prices to {args.prices_output}. Shape: {prices.shape}")
    print(f"Saved daily returns to {args.returns_output}. Shape: {returns.shape}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the main rolling out-of-sample backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio.backtest import BacktestConfig, default_strategy_specs, run_backtest
from portfolio.data import load_frame, save_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", default="data/processed/prices.csv")
    parser.add_argument("--end-date", default="2026-01-31")
    parser.add_argument("--momentum-lookback-days", type=int, default=252)
    parser.add_argument("--momentum-skip-days", type=int, default=21)
    parser.add_argument("--trend-window-days", type=int, default=200)
    parser.add_argument("--covariance-lookback-days", type=int, default=20)
    parser.add_argument("--stress-ewma-span-days", type=int, default=20)
    parser.add_argument("--stress-short-window-days", type=int, default=63)
    parser.add_argument("--stress-reference-window-days", type=int, default=252)
    parser.add_argument("--alpha-scale", type=float, default=0.03)
    parser.add_argument("--target-volatility", type=float, default=0.10)
    parser.add_argument("--transaction-cost-rate", type=float, default=0.001)
    parser.add_argument("--covariance-estimator", default="ewma")
    parser.add_argument("--upper-bound", type=float, default=0.25)
    parser.add_argument("--lambda-tc", type=float, default=0.001)
    parser.add_argument("--eta-l2", type=float, default=0.01)
    parser.add_argument("--rho-mode", choices=["dynamic", "zero", "fixed"], default="dynamic")
    parser.add_argument("--rebalance-frequency", default="M")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = load_frame(PROJECT_ROOT / args.prices)
    if args.end_date:
        prices = prices.loc[: args.end_date]
    config = BacktestConfig(
        momentum_lookback_days=args.momentum_lookback_days,
        momentum_skip_days=args.momentum_skip_days,
        trend_window_days=args.trend_window_days,
        covariance_lookback_days=args.covariance_lookback_days,
        stress_ewma_span_days=args.stress_ewma_span_days,
        stress_short_window_days=args.stress_short_window_days,
        stress_reference_window_days=args.stress_reference_window_days,
        alpha_scale=args.alpha_scale,
        target_volatility=args.target_volatility,
        transaction_cost_rate=args.transaction_cost_rate,
        covariance_estimator=args.covariance_estimator,
        upper_bound=args.upper_bound,
        lambda_tc=args.lambda_tc,
        eta_l2=args.eta_l2,
        rho_mode=args.rho_mode,
        rebalance_frequency=args.rebalance_frequency,
    )
    result = run_backtest(prices, default_strategy_specs(), config)
    save_frame(result.returns, PROJECT_ROOT / "data/processed/backtest_returns.csv")
    save_frame(result.gross_returns, PROJECT_ROOT / "data/processed/backtest_gross_returns.csv")
    result.weights.to_csv(PROJECT_ROOT / "data/processed/backtest_weights.csv", index=False)
    result.diagnostics.to_csv(
        PROJECT_ROOT / "data/processed/backtest_diagnostics.csv", index=False
    )
    result.signals.to_csv(PROJECT_ROOT / "data/processed/backtest_signals.csv", index=False)
    save_frame(result.turnover, PROJECT_ROOT / "data/processed/backtest_turnover.csv")
    save_frame(result.performance, PROJECT_ROOT / "reports/tables/performance_summary.csv")

    print("Backtest complete.")
    print(result.performance.round(4))


if __name__ == "__main__":
    main()

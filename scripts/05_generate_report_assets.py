#!/usr/bin/env python3
"""Generate report figures and derived tables from backtest outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio.data import load_frame
from portfolio.plots import (
    plot_cash_weight,
    plot_cumulative_returns,
    plot_diagnostic_series,
    plot_drawdowns,
    plot_predicted_volatility,
    plot_rolling_sharpe,
    plot_signal_heatmap,
    plot_turnover,
    plot_weights,
)

MAIN_STRATEGY = "Proposed Strategy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returns", default="data/processed/backtest_returns.csv")
    parser.add_argument("--performance", default="reports/tables/performance_summary.csv")
    parser.add_argument("--weights", default="data/processed/backtest_weights.csv")
    parser.add_argument("--turnover", default="data/processed/backtest_turnover.csv")
    parser.add_argument("--diagnostics", default="data/processed/backtest_diagnostics.csv")
    parser.add_argument("--signals", default="data/processed/backtest_signals.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    returns = load_frame(PROJECT_ROOT / args.returns)
    turnover = load_frame(PROJECT_ROOT / args.turnover)
    import pandas as pd

    weights = pd.read_csv(PROJECT_ROOT / args.weights, parse_dates=["date"])
    diagnostics = pd.read_csv(PROJECT_ROOT / args.diagnostics, parse_dates=["date"])
    signals = pd.read_csv(PROJECT_ROOT / args.signals, parse_dates=["date"])
    performance = pd.read_csv(PROJECT_ROOT / args.performance)
    cumulative_plot_returns = _build_cumulative_plot_returns(returns=returns)
    _write_performance_table(performance, PROJECT_ROOT / "reports/tables/performance_summary.tex")
    plot_cumulative_returns(
        cumulative_plot_returns,
        PROJECT_ROOT / "reports/figures/cumulative_returns.png",
        title="Cumulative Net Value: Baselines vs Proposed Strategy",
    )
    plot_drawdowns(cumulative_plot_returns, PROJECT_ROOT / "reports/figures/drawdowns.png")
    plot_rolling_sharpe(returns, PROJECT_ROOT / "reports/figures/rolling_12m_sharpe.png")
    plot_turnover(turnover, PROJECT_ROOT / "reports/figures/turnover.png")
    plot_cash_weight(weights, PROJECT_ROOT / "reports/figures/cash_weight.png")
    plot_weights(
        weights,
        MAIN_STRATEGY,
        PROJECT_ROOT / "reports/figures/cash_allowed_target_vol_weights.png",
    )
    plot_predicted_volatility(
        diagnostics,
        MAIN_STRATEGY,
        PROJECT_ROOT / "reports/figures/predicted_volatility_vs_target.png",
    )
    plot_diagnostic_series(
        diagnostics,
        "v_t",
        MAIN_STRATEGY,
        PROJECT_ROOT / "reports/figures/volatility_stress_ratio.png",
        title="Volatility Stress Ratio",
    )
    plot_diagnostic_series(
        diagnostics,
        "rho_t",
        MAIN_STRATEGY,
        PROJECT_ROOT / "reports/figures/dynamic_rho.png",
        title="Dynamic Robust Penalty",
    )
    plot_signal_heatmap(signals, PROJECT_ROOT / "reports/figures/signal_heatmap.png")
    print("Generated report figures.")


def _build_cumulative_plot_returns(returns):
    """Return baseline and final strategy returns for the cumulative-value figure."""
    columns = ["Equal Weight", "Minimum Variance", "Traditional MVO", MAIN_STRATEGY]
    missing = [column for column in columns if column not in returns.columns]
    if missing:
        raise ValueError(f"Missing final strategy return columns: {missing}")
    return returns.loc[:, columns].dropna(how="any")


def _write_performance_table(performance, path) -> None:
    """Write a compact LaTeX table for the final strategy."""
    table = performance.rename(columns={"Date": "strategy"}).copy()
    columns = [
        "strategy",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "average_turnover",
        "average_cash_weight",
        "average_predicted_volatility",
        "average_rho_t",
        "average_v_t",
        "constraint_violation_count",
    ]
    table = table.loc[:, columns]
    table = table.rename(
        columns={
            "strategy": "Strategy",
            "annualized_return": "Ann. Return",
            "annualized_volatility": "Ann. Vol",
            "sharpe_ratio": "Sharpe",
            "sortino_ratio": "Sortino",
            "max_drawdown": "Max DD",
            "calmar_ratio": "Calmar",
            "average_turnover": "Avg Turnover",
            "average_cash_weight": "Avg Cash",
            "average_predicted_volatility": "Avg Pred Vol",
            "average_rho_t": "Avg Rho",
            "average_v_t": "Avg v",
            "constraint_violation_count": "Violations",
        }
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = table.to_latex(index=False, float_format="%.4f", escape=False)
    output_path.write_text("\\resizebox{\\linewidth}{!}{%\n" + latex + "}\n")


if __name__ == "__main__":
    main()

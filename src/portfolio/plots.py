"""Plotting utilities for report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import cumulative_returns, drawdown, rolling_sharpe


def _prepare_output(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_cumulative_returns(
    returns: pd.DataFrame,
    path: str | Path,
    title: str = "Cumulative Net Value",
) -> None:
    """Save a cumulative wealth figure."""
    output_path = _prepare_output(path)
    ax = cumulative_returns(returns).plot(figsize=(11, 6), linewidth=1.6)
    ax.set_title(title)
    ax.set_ylabel("Growth of 1 USD")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_drawdowns(returns: pd.DataFrame, path: str | Path) -> None:
    """Save a drawdown figure."""
    output_path = _prepare_output(path)
    drawdowns = pd.DataFrame({column: drawdown(returns[column]) for column in returns})
    ax = drawdowns.plot(figsize=(11, 6), linewidth=1.4)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_turnover(turnover: pd.DataFrame, path: str | Path) -> None:
    """Save a turnover-over-time figure."""
    output_path = _prepare_output(path)
    ax = turnover.plot(figsize=(11, 6), linewidth=1.2)
    ax.set_title("Turnover Over Time")
    ax.set_ylabel("One-Way Turnover")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_weights(weights: pd.DataFrame, strategy: str, path: str | Path) -> None:
    """Save a stacked weight chart for one strategy."""
    output_path = _prepare_output(path)
    subset = weights[weights["strategy"] == strategy]
    if subset.empty:
        raise ValueError(f"No weights found for strategy: {strategy}")
    wide = subset.pivot(index="date", columns="asset", values="weight").fillna(0.0)
    ax = wide.plot.area(figsize=(11, 6), linewidth=0.0)
    ax.set_title(f"Portfolio Weights: {strategy}")
    ax.set_ylabel("Weight")
    ax.set_xlabel("")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_cash_weight(weights: pd.DataFrame, path: str | Path, cash_ticker: str = "BIL") -> None:
    """Save cash weight over time for all strategies."""
    output_path = _prepare_output(path)
    subset = weights[weights["asset"] == cash_ticker]
    if subset.empty:
        raise ValueError(f"No cash weights found for asset: {cash_ticker}")
    wide = subset.pivot(index="date", columns="strategy", values="weight").fillna(0.0)
    ax = wide.plot(figsize=(11, 6), linewidth=1.4)
    ax.set_title("Cash Weight Over Time")
    ax.set_ylabel("Cash Weight")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_predicted_volatility(
    diagnostics: pd.DataFrame,
    strategy: str,
    path: str | Path,
) -> None:
    """Save predicted volatility against the target volatility."""
    output_path = _prepare_output(path)
    subset = diagnostics[diagnostics["strategy"] == strategy].copy()
    if subset.empty:
        raise ValueError(f"No diagnostics found for strategy: {strategy}")
    subset = subset.sort_values("date")
    ax = subset.plot(
        x="date",
        y="predicted_volatility",
        figsize=(11, 6),
        linewidth=1.4,
        legend=False,
    )
    if "target_volatility" in subset.columns:
        ax.plot(
            subset["date"],
            subset["target_volatility"],
            linestyle="--",
            linewidth=1.2,
            label="Target Volatility",
        )
        ax.legend(["Predicted Volatility", "Target Volatility"])
    ax.set_title("Predicted Volatility vs Target")
    ax.set_ylabel("Annualized Volatility")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_diagnostic_series(
    diagnostics: pd.DataFrame,
    column: str,
    strategy: str,
    path: str | Path,
    title: str | None = None,
) -> None:
    """Save one diagnostic series for a strategy."""
    output_path = _prepare_output(path)
    subset = diagnostics[diagnostics["strategy"] == strategy].copy()
    if subset.empty or column not in subset.columns:
        raise ValueError(f"No diagnostic column {column!r} found for strategy: {strategy}")
    subset = subset.sort_values("date")
    ax = subset.plot(x="date", y=column, figsize=(11, 6), linewidth=1.4, legend=False)
    ax.set_title(title or column)
    ax.set_ylabel(column)
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_rolling_sharpe(
    returns: pd.DataFrame,
    path: str | Path,
    window: int = 252,
) -> None:
    """Save rolling annualized Sharpe ratios."""
    output_path = _prepare_output(path)
    rolling = returns.apply(lambda series: rolling_sharpe(series, window=window))
    ax = rolling.plot(figsize=(11, 6), linewidth=1.4)
    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_signal_heatmap(scores: pd.DataFrame, path: str | Path) -> None:
    """Save a simple signal heatmap."""
    output_path = _prepare_output(path)
    try:
        import seaborn as sns
    except ImportError as exc:
        raise ImportError("seaborn is required for heatmap plots.") from exc

    if {"date", "asset", "score"}.issubset(scores.columns):
        score_matrix = scores.pivot(index="date", columns="asset", values="score")
    else:
        score_matrix = scores
    plt.figure(figsize=(12, 7))
    sns.heatmap(score_matrix.T, cmap="RdBu_r", center=0.0)
    plt.title("Signal Scores")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

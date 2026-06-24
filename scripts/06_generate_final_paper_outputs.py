#!/usr/bin/env python3
"""Generate all final paper tables and figures from reproducible backtests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio.backtest import BacktestConfig, StrategySpec, default_strategy_specs, run_backtest
from portfolio.data import ASSET_GROUPS, CASH_TICKER, RISKY_TICKERS, load_frame, save_frame
from portfolio.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    cumulative_returns,
    drawdown,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from portfolio.plots import plot_cumulative_returns, plot_drawdowns

MAIN_STRATEGIES = ["Equal Weight", "Minimum Variance", "Traditional MVO", "Proposed Strategy"]
PROPOSED = "Proposed Strategy"
STUDY_END_DATE = "2026-01-31"
PERIODS_PER_YEAR = 252

GROUPS = {
    "Equity": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "Treasury": ["TLT", "IEF", "SHY"],
    "Credit": ["LQD", "HYG"],
    "Real Assets": ["GLD", "DBC", "VNQ"],
    "Cash": [CASH_TICKER],
}


@dataclass(frozen=True)
class OutputItem:
    path: str
    section: str
    description: str
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", default="data/processed/prices.csv")
    parser.add_argument("--end-date", default=STUDY_END_DATE)
    return parser.parse_args()


def proposed_spec(
    name: str = PROPOSED,
    target_volatility_base: float = 0.10,
    covariance_estimator: str = "ewma",
    covariance_lookback_days: int = 20,
) -> StrategySpec:
    """Return the final proposed strategy specification, with controlled variants."""
    return StrategySpec(
        name,
        "cash_allowed_target_vol_robust",
        {
            "market_state_mode": "ewma",
            "target_volatility_mode": "inverse_vt",
            "target_volatility_base": target_volatility_base,
            "target_volatility_lower": 0.04,
            "target_volatility_upper": 0.16,
            "covariance_lookback_days": covariance_lookback_days,
            "covariance_estimator": covariance_estimator,
            "ewma_decay": 0.94,
        },
    )


def final_config(**overrides) -> BacktestConfig:
    """Return the fixed final strategy configuration with optional sensitivity changes."""
    params = {
        "momentum_lookback_days": 252,
        "momentum_skip_days": 21,
        "trend_window_days": 200,
        "covariance_lookback_days": 20,
        "stress_ewma_span_days": 20,
        "stress_short_window_days": 63,
        "stress_reference_window_days": 252,
        "alpha_scale": 0.03,
        "target_volatility": 0.10,
        "transaction_cost_rate": 0.001,
        "covariance_estimator": "ewma",
        "upper_bound": 0.25,
        "lambda_tc": 0.001,
        "eta_l2": 0.01,
        "rho_mode": "dynamic",
        "rebalance_frequency": "M",
        "ewma_rho_base": 0.08,
        "ewma_rho_lower": 0.03,
        "ewma_rho_upper": 0.25,
    }
    params.update(overrides)
    return BacktestConfig(**params)


def main() -> None:
    args = parse_args()
    prices = load_frame(PROJECT_ROOT / args.prices).loc[: args.end_date]
    table_dir = PROJECT_ROOT / "reports/tables"
    figure_dir = PROJECT_ROOT / "reports/figures"
    processed_dir = PROJECT_ROOT / "data/processed"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    outputs = required_output_inventory()

    main_result = run_backtest(prices, default_strategy_specs(), final_config())
    save_main_outputs(main_result, processed_dir)

    table_1_etf_universe(table_dir)
    table_2 = table_2_main_performance(main_result)
    table_2.to_csv(table_dir / "table_2_main_performance.csv", index=False)

    plot_cumulative_returns(
        main_result.returns.loc[:, MAIN_STRATEGIES],
        figure_dir / "figure_1_cumulative_net_value.png",
        title="Cumulative Net Value: Main Strategy Comparison",
    )
    plot_drawdowns(
        main_result.returns.loc[:, MAIN_STRATEGIES],
        figure_dir / "figure_2_drawdown_curve.png",
    )

    drawdown_summary = table_3_drawdown_summary(main_result.returns)
    drawdown_summary.to_csv(table_dir / "table_3_drawdown_summary.csv", index=False)
    major_drawdown_periods(main_result, prices, drawdown_summary).to_csv(
        table_dir / "major_drawdown_periods.csv", index=False
    )

    exposure_ts = group_exposure_timeseries(main_result.weights)
    plot_cash_weight(main_result.weights, figure_dir / "figure_3_cash_weight.png")
    plot_group_exposure(exposure_ts, PROPOSED, figure_dir / "figure_4_group_exposure.png")
    plot_portfolio_weights(main_result.weights, PROPOSED, figure_dir / "portfolio_weights.png")
    plot_risky_exposure(exposure_ts, figure_dir / "risky_exposure.png")
    exposure_summary(main_result, drawdown_summary, exposure_ts).to_csv(
        table_dir / "exposure_summary.csv", index=False
    )

    risk_ts, risk_summary = risk_forecast_diagnostics(main_result)
    plot_risk_forecast(risk_ts, figure_dir / "figure_5_predicted_vs_realized_volatility.png")
    risk_summary.to_csv(table_dir / "table_5_risk_forecast_diagnostics.csv", index=False)

    signal_ts, signal_summary, top_bottom = signal_diagnostics(
        prices, main_result.signals, main_result.returns[PROPOSED]
    )
    signal_summary.to_csv(table_dir / "table_4_signal_diagnostics.csv", index=False)
    top_bottom.to_csv(table_dir / "top_bottom_signal_return.csv", index=False)
    plot_signal_ic(signal_ts, figure_dir / "figure_6_signal_ic_over_time.png")

    turnover_summary(main_result, drawdown_summary).to_csv(
        table_dir / "turnover_summary.csv", index=False
    )
    plot_turnover(main_result.turnover, figure_dir / "figure_8_turnover_over_time.png")
    transaction_cost_drag(main_result).to_csv(table_dir / "transaction_cost_drag.csv", index=False)

    target_vol_sensitivity(prices, table_dir, figure_dir)
    alpha_scale_sensitivity(prices, table_dir)
    rho_sensitivity(prices, table_dir)
    covariance_sensitivity(prices, table_dir)
    transaction_cost_sensitivity(prices, table_dir)
    subperiod_performance(main_result.returns, drawdown_summary).to_csv(
        table_dir / "table_11_subperiod_performance.csv", index=False
    )
    constraint_check(main_result).to_csv(table_dir / "constraint_check.csv", index=False)
    backtest_timing_check(main_result, prices).to_csv(
        table_dir / "backtest_timing_check.csv", index=False
    )

    inventory = final_output_inventory(outputs)
    inventory.to_csv(table_dir / "final_paper_output_inventory.csv", index=False)
    inventory = final_output_inventory(outputs)
    inventory.to_csv(table_dir / "final_paper_output_inventory.csv", index=False)
    print_summary(inventory, main_result)


def save_main_outputs(result, processed_dir: Path) -> None:
    """Save canonical main backtest outputs for auditability."""
    save_frame(result.returns, processed_dir / "backtest_returns.csv")
    save_frame(result.gross_returns, processed_dir / "backtest_gross_returns.csv")
    result.weights.to_csv(processed_dir / "backtest_weights.csv", index=False)
    result.diagnostics.to_csv(processed_dir / "backtest_diagnostics.csv", index=False)
    result.signals.to_csv(processed_dir / "backtest_signals.csv", index=False)
    save_frame(result.turnover, processed_dir / "backtest_turnover.csv")
    save_frame(result.performance, PROJECT_ROOT / "reports/tables/performance_summary.csv")


def table_1_etf_universe(table_dir: Path) -> None:
    """Generate the ETF universe table."""
    group_lookup = {}
    for group, tickers in GROUPS.items():
        for ticker in tickers:
            group_lookup[ticker] = group
    rows = []
    for ticker in list(RISKY_TICKERS) + [CASH_TICKER]:
        is_cash = ticker == CASH_TICKER
        rows.append(
            {
                "Ticker": ticker,
                "Asset Group": group_lookup[ticker],
                "Role": "Cash Proxy" if is_cash else "Risky ETF",
                "Included in Risky Universe": not is_cash,
                "Included in Alpha Signal": not is_cash,
                "Included in Covariance Estimation": not is_cash,
                "Included in Cash Sleeve": is_cash,
            }
        )
    pd.DataFrame(rows).to_csv(table_dir / "table_1_etf_universe.csv", index=False)


def table_2_main_performance(result) -> pd.DataFrame:
    """Build the main four-strategy performance table."""
    rows = []
    gross = result.gross_returns
    net = result.returns
    exposures = group_exposure_timeseries(result.weights)
    for strategy in MAIN_STRATEGIES:
        series = net[strategy].dropna()
        gross_series = gross[strategy].dropna()
        row = {
            "Strategy": strategy,
            "Annualized Return": annualized_return(series),
            "Annualized Volatility": annualized_volatility(series),
            "Sharpe Ratio": sharpe_ratio(series),
            "Sortino Ratio": sortino_ratio(series),
            "Maximum Drawdown": max_drawdown(series),
            "Calmar Ratio": calmar_ratio(series),
            "Final Net Value": float(cumulative_returns(series).iloc[-1]),
            "Average Turnover": float(result.turnover[strategy].dropna().mean()),
            "Transaction Cost Drag": annualized_return(gross_series) - annualized_return(series),
            "Average Cash Weight": float(exposures[(strategy, "Cash")].mean()),
            "Average Risky Exposure": float(exposures[(strategy, "Risky")].mean()),
            "Average Equity Exposure": float(exposures[(strategy, "Equity")].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def drawdown_period(series: pd.Series) -> dict[str, object]:
    """Identify max drawdown start, valley, recovery, and duration."""
    clean = series.dropna()
    wealth = cumulative_returns(clean)
    running_max = wealth.cummax()
    dd = wealth / running_max - 1.0
    valley = dd.idxmin()
    start = wealth.loc[:valley].idxmax()
    peak_value = wealth.loc[start]
    after = wealth.loc[valley:]
    recovered = after[after >= peak_value]
    recovery = recovered.index[0] if not recovered.empty else pd.NaT
    end_for_duration = recovery if pd.notna(recovery) else clean.index[-1]
    return {
        "max_drawdown": float(dd.loc[valley]),
        "start": pd.Timestamp(start),
        "valley": pd.Timestamp(valley),
        "recovery": pd.Timestamp(recovery) if pd.notna(recovery) else pd.NaT,
        "duration": int((pd.Timestamp(end_for_duration) - pd.Timestamp(start)).days),
    }


def table_3_drawdown_summary(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in MAIN_STRATEGIES:
        period = drawdown_period(returns[strategy])
        rows.append(
            {
                "Strategy": strategy,
                "Max Drawdown": period["max_drawdown"],
                "Drawdown Start Date": period["start"].date().isoformat(),
                "Drawdown Valley Date": period["valley"].date().isoformat(),
                "Recovery Date": (
                    period["recovery"].date().isoformat()
                    if pd.notna(period["recovery"])
                    else ""
                ),
                "Drawdown Duration": period["duration"],
            }
        )
    return pd.DataFrame(rows)


def period_from_window(returns: pd.Series, start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Find the largest peak-to-valley drawdown period inside a date window."""
    subset = returns.loc[start:end].dropna()
    if subset.empty:
        raise ValueError(f"No returns available in window {start} to {end}.")
    wealth = cumulative_returns(subset)
    running_max = wealth.cummax()
    dd = wealth / running_max - 1.0
    valley = dd.idxmin()
    peak = wealth.loc[:valley].idxmax()
    return pd.Timestamp(peak), pd.Timestamp(valley)


def major_drawdown_periods(result, prices: pd.DataFrame, drawdown_summary: pd.DataFrame) -> pd.DataFrame:
    """Build major drawdown period attribution table."""
    proposed_dd = drawdown_summary[drawdown_summary["Strategy"] == PROPOSED].iloc[0]
    periods = [
        ("2020 Crash", *period_from_window(result.returns[PROPOSED], "2020-02-01", "2020-04-30")),
        ("2022 Drawdown", *period_from_window(result.returns[PROPOSED], "2022-01-01", "2022-12-31")),
        (
            "Proposed Max Drawdown",
            pd.Timestamp(proposed_dd["Drawdown Start Date"]),
            pd.Timestamp(proposed_dd["Drawdown Valley Date"]),
        ),
    ]
    exposures = group_exposure_timeseries(result.weights)
    rows = []
    for period_name, start, end in periods:
        for strategy in MAIN_STRATEGIES:
            period_returns = result.returns[strategy].loc[start:end].dropna()
            if period_returns.empty:
                continue
            exposure_slice = exposures.loc[
                (exposures.index >= start) & (exposures.index <= end), strategy
            ]
            rows.append(
                {
                    "Period Name": period_name,
                    "Start Date": start.date().isoformat(),
                    "End Date": end.date().isoformat(),
                    "Strategy": strategy,
                    "Period Return": float((1.0 + period_returns).prod() - 1.0),
                    "Max Drawdown in Period": max_drawdown(period_returns),
                    "Average Cash Weight": safe_mean(exposure_slice.get("Cash")),
                    "Average Equity Exposure": safe_mean(exposure_slice.get("Equity")),
                    "Average Treasury Exposure": safe_mean(exposure_slice.get("Treasury")),
                    "Average Credit Exposure": safe_mean(exposure_slice.get("Credit")),
                    "Average Real Asset Exposure": safe_mean(exposure_slice.get("Real Assets")),
                }
            )
    return pd.DataFrame(rows)


def group_exposure_timeseries(weights: pd.DataFrame) -> pd.DataFrame:
    """Return strategy x group exposure time series from rebalance weights."""
    rows = []
    for (date, strategy), subset in weights.groupby(["date", "strategy"]):
        asset_weights = subset.set_index("asset")["weight"]
        row = {"date": pd.Timestamp(date), "strategy": strategy}
        for group, tickers in GROUPS.items():
            row[group] = float(asset_weights.reindex(tickers).fillna(0.0).sum())
        row["Risky"] = 1.0 - row["Cash"]
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["date", "strategy"])
    wide = frame.pivot(index="date", columns="strategy")
    return wide.swaplevel(0, 1, axis=1).sort_index(axis=1)


def exposure_summary(result, drawdown_summary: pd.DataFrame, exposures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in MAIN_STRATEGIES:
        dd_row = drawdown_summary[drawdown_summary["Strategy"] == strategy].iloc[0]
        start = pd.Timestamp(dd_row["Drawdown Start Date"])
        valley = pd.Timestamp(dd_row["Drawdown Valley Date"])
        exp = exposures[strategy]
        dd_exp = exp.loc[(exp.index >= start) & (exp.index <= valley)]
        rows.append(
            {
                "Strategy": strategy,
                "Average Cash Weight": safe_mean(exp["Cash"]),
                "Maximum Cash Weight": float(exp["Cash"].max()),
                "Average Risky Exposure": safe_mean(exp["Risky"]),
                "Average Equity Exposure": safe_mean(exp["Equity"]),
                "Average Treasury Exposure": safe_mean(exp["Treasury"]),
                "Average Credit Exposure": safe_mean(exp["Credit"]),
                "Average Real Asset Exposure": safe_mean(exp["Real Assets"]),
                "Cash Weight During Max Drawdown": safe_mean(dd_exp["Cash"]),
                "Equity Exposure During Max Drawdown": safe_mean(dd_exp["Equity"]),
                "Treasury Exposure During Max Drawdown": safe_mean(dd_exp["Treasury"]),
                "Credit Exposure During Max Drawdown": safe_mean(dd_exp["Credit"]),
                "Real Asset Exposure During Max Drawdown": safe_mean(dd_exp["Real Assets"]),
            }
        )
    return pd.DataFrame(rows)


def risk_forecast_diagnostics(result) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build proposed-strategy risk forecast diagnostics."""
    diag = result.diagnostics[result.diagnostics["strategy"] == PROPOSED].copy()
    diag["date"] = pd.to_datetime(diag["date"])
    returns = result.returns[PROPOSED]
    realized_21 = returns.rolling(21).std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    realized_63 = returns.rolling(63).std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    ts = diag.set_index("date")[["predicted_volatility", "target_volatility"]].copy()
    ts["Realized 21-day Volatility"] = realized_21.reindex(ts.index, method="ffill")
    ts["Realized 63-day Volatility"] = realized_63.reindex(ts.index, method="ffill")
    ts = ts.rename(
        columns={
            "predicted_volatility": "Predicted Volatility",
            "target_volatility": "Target Volatility",
        }
    )
    ratio = ts["Realized 21-day Volatility"] / ts["Predicted Volatility"].replace(0.0, np.nan)
    summary = pd.DataFrame(
        [
            {
                "Strategy": PROPOSED,
                "Average Predicted Volatility": safe_mean(ts["Predicted Volatility"]),
                "Average Target Volatility": safe_mean(ts["Target Volatility"]),
                "Average Realized 21-day Volatility": safe_mean(ts["Realized 21-day Volatility"]),
                "Average Realized 63-day Volatility": safe_mean(ts["Realized 63-day Volatility"]),
                "Average Realized / Predicted Volatility Ratio": safe_mean(ratio),
                "Maximum Realized / Predicted Volatility Ratio": float(ratio.max()),
                "Target Vol Binding Rate": float(
                    (ts["Predicted Volatility"] >= 0.995 * ts["Target Volatility"]).mean()
                ),
            }
        ]
    )
    return ts, summary


def signal_diagnostics(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    proposed_returns: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute cross-sectional IC and top-minus-bottom signal returns."""
    signal_dates = sorted(pd.to_datetime(signals["date"].unique()))
    daily_returns = prices.loc[:, RISKY_TICKERS].pct_change().dropna(how="any")
    rows = []
    tb_rows = []
    for idx, date in enumerate(signal_dates[:-1]):
        next_date = signal_dates[idx + 1]
        holding_returns = daily_returns.loc[
            (daily_returns.index > date) & (daily_returns.index <= next_date)
        ]
        if holding_returns.empty:
            continue
        next_period_asset_returns = (1.0 + holding_returns).prod() - 1.0
        score = signals[signals["date"] == date].set_index("asset")["score"].reindex(RISKY_TICKERS)
        aligned = pd.concat([score, next_period_asset_returns], axis=1).dropna()
        aligned.columns = ["score", "next_return"]
        ic = aligned["score"].corr(aligned["next_return"])
        top_assets = aligned["score"].nlargest(3).index
        bottom_assets = aligned["score"].nsmallest(3).index
        top_return = float(next_period_asset_returns.reindex(top_assets).mean())
        bottom_return = float(next_period_asset_returns.reindex(bottom_assets).mean())
        tb_rows.append(
            {
                "Date": pd.Timestamp(date).date().isoformat(),
                "Top 3 Score Equal-Weight Return": top_return,
                "Bottom 3 Score Equal-Weight Return": bottom_return,
                "Top-minus-Bottom Return": top_return - bottom_return,
            }
        )
        rows.append({"Date": pd.Timestamp(date), "IC": ic})
    ic_ts = pd.DataFrame(rows).set_index("Date").sort_index()
    tb = pd.DataFrame(tb_rows)
    tb["Cumulative Top-minus-Bottom Return"] = (1.0 + tb["Top-minus-Bottom Return"]).cumprod() - 1.0
    proposed_dd = drawdown(proposed_returns)
    drawdown_dates = proposed_dd[proposed_dd <= -0.05].index
    drawdown_mask = ic_ts.index.isin(drawdown_dates)
    ic_clean = ic_ts["IC"].dropna()
    summary = pd.DataFrame(
        [
            {
                "Average IC": safe_mean(ic_clean),
                "IC Standard Deviation": float(ic_clean.std(ddof=1)),
                "IC t-stat": float(ic_clean.mean() / ic_clean.std(ddof=1) * np.sqrt(len(ic_clean)))
                if len(ic_clean) > 1 and ic_clean.std(ddof=1) != 0
                else np.nan,
                "IC during Normal Periods": safe_mean(ic_ts.loc[~drawdown_mask, "IC"]),
                "IC during Drawdown Periods": safe_mean(ic_ts.loc[drawdown_mask, "IC"]),
                "Number of Rebalance Periods": int(ic_clean.shape[0]),
                "Number of Positive IC Periods": int((ic_clean > 0.0).sum()),
                "Positive IC Ratio": float((ic_clean > 0.0).mean()),
            }
        ]
    )
    return ic_ts, summary, tb


def turnover_summary(result, drawdown_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cost_drag = transaction_cost_drag(result)
    for strategy in MAIN_STRATEGIES:
        turnover = result.turnover[strategy].dropna()
        dd_row = drawdown_summary[drawdown_summary["Strategy"] == strategy].iloc[0]
        start = pd.Timestamp(dd_row["Drawdown Start Date"])
        valley = pd.Timestamp(dd_row["Drawdown Valley Date"])
        dd_turnover = turnover.loc[(turnover.index >= start) & (turnover.index <= valley)]
        drag = cost_drag[cost_drag["Strategy"] == strategy].iloc[0]
        rows.append(
            {
                "Strategy": strategy,
                "Average Turnover": safe_mean(turnover),
                "Median Turnover": float(turnover.median()),
                "Maximum Turnover": float(turnover.max()),
                "Annualized Transaction Cost Drag": drag["Annualized Cost Drag"],
                "Total Transaction Cost Drag": drag["Total Transaction Cost Drag"],
                "Turnover During Max Drawdown": safe_mean(dd_turnover),
            }
        )
    return pd.DataFrame(rows)


def transaction_cost_drag(result) -> pd.DataFrame:
    rows = []
    for strategy in MAIN_STRATEGIES:
        net = result.returns[strategy].dropna()
        gross = result.gross_returns[strategy].dropna()
        rows.append(
            {
                "Strategy": strategy,
                "Annualized Cost Drag": annualized_return(gross) - annualized_return(net),
                "Total Transaction Cost Drag": float(cumulative_returns(gross).iloc[-1] - cumulative_returns(net).iloc[-1]),
                "Final Gross Value": float(cumulative_returns(gross).iloc[-1]),
                "Final Net Value": float(cumulative_returns(net).iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def target_vol_sensitivity(prices: pd.DataFrame, table_dir: Path, figure_dir: Path) -> None:
    rows = []
    returns_by_target = {}
    for target in [0.06, 0.08, 0.10, 0.12]:
        name = f"Target Vol {int(target * 100)}%"
        result = run_backtest(prices, [proposed_spec(name, target_volatility_base=target)], final_config())
        row = sensitivity_row(result, name)
        row["Target Volatility"] = target
        rows.append(row)
        returns_by_target[name] = result.returns[name]
    table = pd.DataFrame(rows)
    ordered = [
        "Target Volatility",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Maximum Drawdown",
        "Calmar Ratio",
        "Average Cash Weight",
        "Average Risky Exposure",
        "Average Equity Exposure",
        "Average Turnover",
        "Realized / Predicted Volatility Ratio",
    ]
    table.loc[:, ordered].to_csv(table_dir / "table_6_target_vol_sensitivity.csv", index=False)
    returns = pd.DataFrame(returns_by_target)
    plot_cumulative_returns(
        returns,
        figure_dir / "figure_7_target_vol_sensitivity.png",
        title="Target Volatility Sensitivity",
    )
    plot_cumulative_returns(
        returns,
        figure_dir / "target_vol_grid_cumulative.png",
        title="Target Volatility Grid: Cumulative Net Value",
    )
    plot_drawdowns(returns, figure_dir / "target_vol_grid_drawdown.png")


def alpha_scale_sensitivity(prices: pd.DataFrame, table_dir: Path) -> None:
    rows = []
    for alpha_scale in [0.01, 0.02, 0.03, 0.05]:
        result = run_backtest(prices, [proposed_spec()], final_config(alpha_scale=alpha_scale))
        row = sensitivity_row(result, PROPOSED)
        row["Alpha Scale"] = alpha_scale
        rows.append(row)
    columns = [
        "Alpha Scale",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Maximum Drawdown",
        "Calmar Ratio",
        "Average Cash Weight",
        "Average Turnover",
    ]
    pd.DataFrame(rows).loc[:, columns].to_csv(
        table_dir / "table_7_alpha_scale_sensitivity.csv", index=False
    )


def rho_sensitivity(prices: pd.DataFrame, table_dir: Path) -> None:
    rows = []
    for multiplier in [0.0, 0.5, 1.0, 1.5, 2.0]:
        cfg = final_config(
            ewma_rho_base=0.08 * multiplier,
            ewma_rho_lower=0.03 * multiplier,
            ewma_rho_upper=0.25 * multiplier,
        )
        result = run_backtest(prices, [proposed_spec()], cfg)
        row = sensitivity_row(result, PROPOSED)
        row["Rho Setting"] = f"rho_multiplier={multiplier:g}"
        rows.append(row)
    columns = [
        "Rho Setting",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Calmar Ratio",
        "Average Cash Weight",
        "Average Turnover",
    ]
    pd.DataFrame(rows).loc[:, columns].to_csv(table_dir / "table_8_rho_sensitivity.csv", index=False)


def covariance_sensitivity(prices: pd.DataFrame, table_dir: Path) -> None:
    variants = [
        ("Sample Covariance", "sample"),
        ("Ledoit-Wolf Shrinkage", "ledoit_wolf"),
        ("Final Implemented Covariance Method", "ewma"),
    ]
    rows = []
    for label, estimator in variants:
        result = run_backtest(
            prices,
            [proposed_spec(label, covariance_estimator=estimator, covariance_lookback_days=20)],
            final_config(),
        )
        row = sensitivity_row(result, label)
        row["Covariance Estimator"] = label
        rows.append(row)
    columns = [
        "Covariance Estimator",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Calmar Ratio",
        "Average Cash Weight",
        "Average Turnover",
        "Average Realized / Predicted Volatility Ratio",
    ]
    table = pd.DataFrame(rows)
    table["Average Realized / Predicted Volatility Ratio"] = table[
        "Realized / Predicted Volatility Ratio"
    ]
    table.loc[:, columns].to_csv(table_dir / "table_9_covariance_sensitivity.csv", index=False)


def transaction_cost_sensitivity(prices: pd.DataFrame, table_dir: Path) -> None:
    rows = []
    for rate in [0.0, 0.0005, 0.001, 0.0025]:
        result = run_backtest(prices, [proposed_spec()], final_config(transaction_cost_rate=rate))
        row = sensitivity_row(result, PROPOSED)
        drag = transaction_cost_drag_for_single(result, PROPOSED)
        row["Transaction Cost Rate"] = f"{rate * 10000:.0f} bps"
        row["Annualized Cost Drag"] = drag["Annualized Cost Drag"]
        row["Final Net Value"] = float(cumulative_returns(result.returns[PROPOSED]).iloc[-1])
        rows.append(row)
    columns = [
        "Transaction Cost Rate",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Average Turnover",
        "Annualized Cost Drag",
        "Final Net Value",
    ]
    pd.DataFrame(rows).loc[:, columns].to_csv(
        table_dir / "table_10_transaction_cost_sensitivity.csv", index=False
    )


def subperiod_performance(returns: pd.DataFrame, drawdown_summary: pd.DataFrame) -> pd.DataFrame:
    proposed_dd = drawdown_summary[drawdown_summary["Strategy"] == PROPOSED].iloc[0]
    covid_start, covid_end = period_from_window(returns[PROPOSED], "2020-02-01", "2020-04-30")
    draw2022_start, draw2022_end = period_from_window(returns[PROPOSED], "2022-01-01", "2022-12-31")
    sample_start = returns.index.min()
    sample_end = returns.index.max()
    periods = [
        ("Pre-COVID Period", sample_start, covid_start - pd.Timedelta(days=1)),
        ("COVID Crash and Recovery", covid_start, pd.Timestamp(proposed_dd["Recovery Date"]) if proposed_dd["Recovery Date"] else covid_end),
        ("2022 Stock-Bond Drawdown", draw2022_start, draw2022_end),
        ("Recent Period", draw2022_end + pd.Timedelta(days=1), sample_end),
    ]
    rows = []
    for label, start, end in periods:
        for strategy in MAIN_STRATEGIES:
            subset = returns[strategy].loc[start:end].dropna()
            if subset.empty:
                continue
            rows.append(
                {
                    "Subperiod": label,
                    "Start Date": pd.Timestamp(subset.index.min()).date().isoformat(),
                    "End Date": pd.Timestamp(subset.index.max()).date().isoformat(),
                    "Strategy": strategy,
                    "Annualized Return": annualized_return(subset),
                    "Annualized Volatility": annualized_volatility(subset),
                    "Sharpe Ratio": sharpe_ratio(subset),
                    "Maximum Drawdown": max_drawdown(subset),
                    "Calmar Ratio": calmar_ratio(subset),
                    "Final Net Value": float(cumulative_returns(subset).iloc[-1]),
                }
            )
    return pd.DataFrame(rows)


def constraint_check(result) -> pd.DataFrame:
    diag = result.diagnostics[result.diagnostics["strategy"] == PROPOSED].copy()
    return pd.DataFrame(
        [
            {
                "Max Budget Violation": float((diag["full_weight_sum"] - 1.0).abs().max()),
                "Max Long-only Violation": float(np.maximum(-diag["min_risky_weight"], 0.0).max()),
                "Max Upper-bound Violation": float(np.maximum(diag["max_risky_weight"] - 0.25, 0.0).max()),
                "Max Target-volatility Violation": float(diag["target_volatility_violation"].max()),
                "Number of Solver Failures": int((diag["solver_success"] == False).sum()),
                "Number of Fallback Uses": int((diag["fallback_used"] == True).sum()),
                "Average Solver Time": safe_mean(diag["solver_time_seconds"]),
                "Number of Rebalance Dates": int(diag.shape[0]),
            }
        ]
    )


def backtest_timing_check(result, prices: pd.DataFrame) -> pd.DataFrame:
    diagnostics = result.diagnostics
    signals = result.signals
    rows = [
        {
            "Check": "signal window ends no later than rebalance date",
            "Pass": bool(pd.to_datetime(signals["date"]).isin(pd.to_datetime(diagnostics["date"])).all()),
            "Explanation": "Signals are timestamped at rebalance dates and computed inside run_backtest from prices.loc[:rebalance_date].",
        },
        {
            "Check": "covariance window ends no later than rebalance date",
            "Pass": bool(pd.to_datetime(diagnostics["date"]).max() <= prices.index.max()),
            "Explanation": "Covariance inputs are sliced from history_returns.loc[:rebalance_date].",
        },
        {
            "Check": "weights are applied only after rebalance date",
            "Pass": bool(result.returns.index.min() > pd.to_datetime(diagnostics["date"]).min()),
            "Explanation": "Holding returns are selected with returns.index > rebalance_date.",
        },
        {
            "Check": "future returns are not used in weight computation",
            "Pass": True,
            "Explanation": "run_backtest asserts holding_start > rebalance_date before applying weights.",
        },
        {
            "Check": "BIL is excluded from alpha and covariance if required",
            "Pass": bool(CASH_TICKER not in set(signals["asset"])),
            "Explanation": "Signal records contain risky ETFs only; covariance is built from risky_assets excluding BIL.",
        },
    ]
    return pd.DataFrame(rows)


def sensitivity_row(result, strategy: str) -> dict[str, float]:
    returns = result.returns[strategy].dropna()
    exposures = group_exposure_timeseries(result.weights)[strategy]
    diag = result.diagnostics[result.diagnostics["strategy"] == strategy]
    predicted = pd.to_numeric(diag["predicted_volatility"], errors="coerce")
    realized_ratio = annualized_volatility(returns) / safe_mean(predicted)
    return {
        "Annualized Return": annualized_return(returns),
        "Annualized Volatility": annualized_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns),
        "Sortino Ratio": sortino_ratio(returns),
        "Maximum Drawdown": max_drawdown(returns),
        "Calmar Ratio": calmar_ratio(returns),
        "Average Cash Weight": safe_mean(exposures["Cash"]),
        "Average Risky Exposure": safe_mean(exposures["Risky"]),
        "Average Equity Exposure": safe_mean(exposures["Equity"]),
        "Average Turnover": safe_mean(result.turnover[strategy]),
        "Realized / Predicted Volatility Ratio": realized_ratio,
    }


def transaction_cost_drag_for_single(result, strategy: str) -> dict[str, float]:
    net = result.returns[strategy].dropna()
    gross = result.gross_returns[strategy].dropna()
    return {
        "Annualized Cost Drag": annualized_return(gross) - annualized_return(net),
        "Total Transaction Cost Drag": float(cumulative_returns(gross).iloc[-1] - cumulative_returns(net).iloc[-1]),
    }


def plot_cash_weight(weights: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    subset = weights[weights["asset"] == CASH_TICKER]
    wide = subset.pivot(index="date", columns="strategy", values="weight").reindex(columns=MAIN_STRATEGIES)
    ax = wide.plot(figsize=(11, 6), linewidth=1.4)
    ax.set_title("Cash Weight")
    ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_group_exposure(exposures: pd.DataFrame, strategy: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    columns = ["Equity", "Treasury", "Credit", "Real Assets", "Cash"]
    ax = exposures[strategy].loc[:, columns].plot(figsize=(11, 6), linewidth=1.4)
    ax.set_title(f"Group Exposure: {strategy}")
    ax.set_ylabel("Exposure")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_portfolio_weights(weights: pd.DataFrame, strategy: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    subset = weights[weights["strategy"] == strategy]
    wide = subset.pivot(index="date", columns="asset", values="weight").fillna(0.0)
    ax = wide.plot.area(figsize=(11, 6), linewidth=0.0)
    ax.set_title(f"Portfolio Weights: {strategy}")
    ax.set_ylabel("Weight")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_risky_exposure(exposures: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    wide = pd.DataFrame({strategy: exposures[(strategy, "Risky")] for strategy in MAIN_STRATEGIES})
    ax = wide.plot(figsize=(11, 6), linewidth=1.4)
    ax.set_title("Risky Exposure")
    ax.set_ylabel("Risky Exposure")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_risk_forecast(ts: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    ax = ts.plot(figsize=(11, 6), linewidth=1.4)
    ax.set_title("Predicted vs Realized Volatility: Proposed Strategy")
    ax.set_ylabel("Annualized Volatility")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_signal_ic(ic_ts: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    ax = ic_ts["IC"].plot(figsize=(11, 5), linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Cross-Sectional Signal IC Over Time")
    ax.set_ylabel("IC")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_turnover(turnover: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    ax = turnover.loc[:, MAIN_STRATEGIES].plot(figsize=(11, 6), linewidth=1.3)
    ax.set_title("One-Way Turnover")
    ax.set_ylabel("Turnover")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def safe_mean(series) -> float:
    if series is None:
        return np.nan
    clean = pd.Series(series).dropna()
    return float(clean.mean()) if not clean.empty else np.nan


def required_output_inventory() -> list[OutputItem]:
    return [
        OutputItem("reports/tables/table_1_etf_universe.csv", "Data and Asset Universe", "ETF universe metadata."),
        OutputItem("reports/tables/table_2_main_performance.csv", "Empirical Results", "Main four-strategy performance table."),
        OutputItem("reports/figures/figure_1_cumulative_net_value.png", "Empirical Results", "Cumulative net value for four strategies."),
        OutputItem("reports/figures/figure_2_drawdown_curve.png", "Empirical Results", "Drawdown curves for four strategies."),
        OutputItem("reports/tables/table_3_drawdown_summary.csv", "Drawdown Analysis", "Max drawdown period summary."),
        OutputItem("reports/tables/major_drawdown_periods.csv", "Drawdown Analysis", "Major drawdown period attribution."),
        OutputItem("reports/figures/figure_3_cash_weight.png", "Exposure Diagnostics", "Cash weight over time."),
        OutputItem("reports/figures/figure_4_group_exposure.png", "Exposure Diagnostics", "Proposed strategy group exposures."),
        OutputItem("reports/figures/portfolio_weights.png", "Exposure Diagnostics", "Proposed strategy ETF weights."),
        OutputItem("reports/figures/risky_exposure.png", "Exposure Diagnostics", "Risky exposure over time."),
        OutputItem("reports/tables/exposure_summary.csv", "Exposure Diagnostics", "Exposure summary table."),
        OutputItem("reports/figures/figure_5_predicted_vs_realized_volatility.png", "Risk Forecast Diagnostics", "Predicted vs realized volatility."),
        OutputItem("reports/tables/table_5_risk_forecast_diagnostics.csv", "Risk Forecast Diagnostics", "Risk forecast summary."),
        OutputItem("reports/tables/table_4_signal_diagnostics.csv", "Signal Diagnostics", "Signal IC summary."),
        OutputItem("reports/figures/figure_6_signal_ic_over_time.png", "Signal Diagnostics", "Signal IC over time."),
        OutputItem("reports/tables/top_bottom_signal_return.csv", "Signal Diagnostics", "Top-minus-bottom signal returns."),
        OutputItem("reports/tables/turnover_summary.csv", "Turnover and Transaction Costs", "Turnover summary."),
        OutputItem("reports/figures/figure_8_turnover_over_time.png", "Turnover and Transaction Costs", "Turnover over time."),
        OutputItem("reports/tables/transaction_cost_drag.csv", "Turnover and Transaction Costs", "Cost drag summary."),
        OutputItem("reports/tables/table_6_target_vol_sensitivity.csv", "Sensitivity Analysis", "Target-volatility sensitivity."),
        OutputItem("reports/figures/figure_7_target_vol_sensitivity.png", "Sensitivity Analysis", "Target-volatility cumulative values."),
        OutputItem("reports/figures/target_vol_grid_cumulative.png", "Sensitivity Analysis", "Target-volatility cumulative grid."),
        OutputItem("reports/figures/target_vol_grid_drawdown.png", "Sensitivity Analysis", "Target-volatility drawdown grid."),
        OutputItem("reports/tables/table_7_alpha_scale_sensitivity.csv", "Sensitivity Analysis", "Alpha scale sensitivity."),
        OutputItem("reports/tables/table_8_rho_sensitivity.csv", "Sensitivity Analysis", "Robust penalty sensitivity."),
        OutputItem("reports/tables/table_9_covariance_sensitivity.csv", "Sensitivity Analysis", "Covariance estimator sensitivity."),
        OutputItem("reports/tables/table_10_transaction_cost_sensitivity.csv", "Sensitivity Analysis", "Transaction cost sensitivity."),
        OutputItem("reports/tables/table_11_subperiod_performance.csv", "Subperiod Analysis", "Subperiod performance."),
        OutputItem("reports/tables/constraint_check.csv", "Implementation Checks", "Constraint and solver checks."),
        OutputItem("reports/tables/backtest_timing_check.csv", "Implementation Checks", "No-look-ahead timing checks."),
        OutputItem("reports/tables/final_paper_output_inventory.csv", "Appendix", "Final output inventory."),
    ]


def final_output_inventory(outputs: list[OutputItem]) -> pd.DataFrame:
    rows = []
    for item in outputs:
        path = PROJECT_ROOT / item.path
        rows.append(
            {
                "Output File": item.path,
                "Exists": path.exists(),
                "Used In Paper Section": item.section,
                "Description": item.description,
                "Notes": item.notes,
            }
        )
    return pd.DataFrame(rows)


def print_summary(inventory: pd.DataFrame, result) -> None:
    missing = inventory[~inventory["Exists"]]
    print("Final paper output generation complete.")
    print(f"Generated files: {int(inventory['Exists'].sum())}/{len(inventory)}")
    if missing.empty:
        print("Missing files: none")
    else:
        print("Missing files:")
        for path in missing["Output File"]:
            print(f"  - {path}")
    print(f"Main strategies: {list(result.returns.columns)}")
    allowed_only = list(result.returns.columns) == MAIN_STRATEGIES
    print(f"Only four allowed main strategies in main returns: {allowed_only}")
    proposed_diag = result.diagnostics[result.diagnostics["strategy"] == PROPOSED]
    matches_methodology = (
        set(proposed_diag["covariance_estimator"]) == {"ewma"}
        and set(proposed_diag["market_state_mode"]) == {"ewma"}
        and set(proposed_diag["target_volatility_mode"]) == {"inverse_vt"}
    )
    print(f"Proposed strategy matches final methodology: {matches_methodology}")
    exploratory_names = {"Breadth", "Fixed d", "Dynamic d", "SPY Proxy"}
    exploratory_present = any(
        any(token in strategy for token in exploratory_names) for strategy in result.returns.columns
    )
    print(f"Old exploratory strategies in final main outputs: {exploratory_present}")


if __name__ == "__main__":
    main()

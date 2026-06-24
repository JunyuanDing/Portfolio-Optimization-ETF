"""Rolling out-of-sample backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .costs import one_way_turnover, simulate_holding_period, transaction_cost
from .covariance import estimate_covariance_with_diagnostics
from .data import CASH_TICKER, RISKY_TICKERS
from .metrics import performance_summary
from .parameters import compute_market_state
from .robust_optimization import (
    CashAllowedOptimizationConfig,
    equal_weight_risky,
    full_weight_vector,
    solve_cash_allowed_target_vol_robust,
    solve_minimum_variance_long_only,
    solve_traditional_mvo_long_only,
)
from .signals import SignalConfig, alpha_from_scores, signal_at_date
from .validation import assert_excludes_cash, assert_fixed_lambda_tc, assert_no_lookahead, validate_cash_allowed_weights

StrategyKind = Literal[
    "equal_weight",
    "minimum_variance",
    "traditional_mvo",
    "cash_allowed_target_vol_robust",
]
RhoMode = Literal["dynamic", "zero", "fixed"]
MarketStateMode = Literal["standard", "ewma"]
TargetVolatilityMode = Literal["fixed", "inverse_vt"]


@dataclass(frozen=True)
class StrategySpec:
    """Strategy definition used by the backtest engine."""

    name: str
    kind: StrategyKind
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest configuration for the final strategy."""

    momentum_lookback_days: int = 252
    momentum_skip_days: int = 21
    trend_window_days: int = 200
    covariance_lookback_days: int = 252
    stress_ewma_span_days: int = 20
    stress_short_window_days: int = 63
    stress_reference_window_days: int = 252
    periods_per_year: int = 252
    transaction_cost_rate: float = 0.001
    rebalance_frequency: str = "M"
    alpha_scale: float = 0.03
    signal_mode: str = "both"
    covariance_estimator: str = "ledoit_wolf"
    cash_ticker: str = CASH_TICKER
    risky_tickers: tuple[str, ...] = tuple(RISKY_TICKERS)
    upper_bound: float = 0.25
    target_volatility: float = 0.10
    lambda_tc: float = 0.001
    eta_l2: float = 0.01
    rho_mode: RhoMode = "dynamic"
    fixed_rho: float = 0.05
    ewma_rho_base: float = 0.08
    ewma_rho_lower: float = 0.03
    ewma_rho_upper: float = 0.25


@dataclass
class BacktestResult:
    """Backtest outputs."""

    returns: pd.DataFrame
    gross_returns: pd.DataFrame
    weights: pd.DataFrame
    diagnostics: pd.DataFrame
    turnover: pd.DataFrame
    signals: pd.DataFrame
    performance: pd.DataFrame


def default_strategy_specs() -> list[StrategySpec]:
    """Return the final comparison set used in the paper."""
    return [
        StrategySpec("Equal Weight", "equal_weight"),
        StrategySpec(
            "Minimum Variance",
            "minimum_variance",
            {"covariance_lookback_days": 252, "covariance_estimator": "sample"},
        ),
        StrategySpec(
            "Traditional MVO",
            "traditional_mvo",
            {
                "covariance_lookback_days": 252,
                "covariance_estimator": "sample",
                "risk_aversion": 1.0,
            },
        ),
        StrategySpec(
            "Proposed Strategy",
            "cash_allowed_target_vol_robust",
            {
                "market_state_mode": "ewma",
                "target_volatility_mode": "inverse_vt",
                "target_volatility_base": 0.10,
                "target_volatility_lower": 0.04,
                "target_volatility_upper": 0.16,
                "covariance_lookback_days": 20,
                "covariance_estimator": "ewma",
                "ewma_decay": 0.94,
            },
        ),
    ]


def monthly_rebalance_dates(
    returns: pd.DataFrame,
    lookback_days: int,
    frequency: str = "M",
) -> list[pd.Timestamp]:
    """Select period-end rebalance dates after the initial lookback window."""
    eligible = returns.iloc[lookback_days:]
    if eligible.empty:
        raise ValueError("Not enough returns for the requested lookback window.")
    date_series = pd.Series(eligible.index, index=eligible.index)
    dates = date_series.groupby(date_series.index.to_period(frequency)).max().tolist()
    return [pd.Timestamp(date) for date in dates]


def required_history_days(config: BacktestConfig) -> int:
    """Return the minimum return-history warm-up needed before rebalancing."""
    return max(
        config.momentum_lookback_days,
        config.trend_window_days,
        config.covariance_lookback_days,
        config.stress_ewma_span_days,
        config.stress_reference_window_days,
    )


def run_backtest(
    prices: pd.DataFrame,
    strategies: list[StrategySpec] | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a strictly out-of-sample rolling backtest."""
    cfg = config or BacktestConfig()
    assert_fixed_lambda_tc(cfg.lambda_tc)
    strategy_specs = strategies or default_strategy_specs()
    returns = prices.pct_change().dropna(how="any")
    risky_assets = [ticker for ticker in cfg.risky_tickers if ticker in returns.columns]
    full_assets = risky_assets + [cfg.cash_ticker]
    if cfg.cash_ticker not in returns.columns:
        raise ValueError(f"Cash ticker {cfg.cash_ticker} is required in returns.")
    if len(risky_assets) != len(cfg.risky_tickers):
        missing = sorted(set(cfg.risky_tickers) - set(risky_assets))
        raise ValueError(f"Missing risky assets: {missing}")
    assert_excludes_cash(risky_assets, cfg.cash_ticker, "risky_assets")

    returns = returns.loc[:, full_assets]
    prices = prices.loc[:, full_assets]
    warmup_days = required_history_days(cfg)
    rebalance_dates = monthly_rebalance_dates(
        returns.loc[:, risky_assets], warmup_days, cfg.rebalance_frequency
    )
    if len(rebalance_dates) < 2:
        raise ValueError("At least two rebalance dates are required.")

    initial_cash = pd.Series(0.0, index=full_assets, dtype=float)
    initial_cash.loc[cfg.cash_ticker] = 1.0
    current_weights = {spec.name: initial_cash.copy() for spec in strategy_specs}
    daily_net_returns_by_strategy = {spec.name: [] for spec in strategy_specs}
    daily_gross_returns_by_strategy = {spec.name: [] for spec in strategy_specs}
    weights_records: list[dict[str, Any]] = []
    diagnostics_records: list[dict[str, Any]] = []
    turnover_records: list[dict[str, Any]] = []
    signal_records: list[dict[str, Any]] = []

    for idx, rebalance_date in enumerate(rebalance_dates[:-1]):
        next_rebalance_date = rebalance_dates[idx + 1]
        history_returns = returns.loc[:rebalance_date, full_assets]
        base_covariance_returns_window = history_returns.loc[:, risky_assets].tail(
            cfg.covariance_lookback_days
        )
        stress_returns_window = history_returns.loc[:, risky_assets].tail(
            cfg.stress_reference_window_days
        )
        if (
            len(base_covariance_returns_window) < cfg.covariance_lookback_days
            or len(stress_returns_window) < cfg.stress_reference_window_days
        ):
            continue
        holding_returns = returns.loc[
            (returns.index > rebalance_date) & (returns.index <= next_rebalance_date),
            full_assets,
        ]
        if holding_returns.empty:
            continue
        assert_no_lookahead(rebalance_date, holding_returns.index[0])

        signal_cfg = SignalConfig(
            risky_tickers=tuple(risky_assets),
            cash_ticker=cfg.cash_ticker,
            momentum_lookback_days=cfg.momentum_lookback_days,
            momentum_skip_days=cfg.momentum_skip_days,
            trend_window_days=cfg.trend_window_days,
            alpha_scale=cfg.alpha_scale,
            mode=cfg.signal_mode,  # type: ignore[arg-type]
        )
        scores = signal_at_date(prices.loc[:, risky_assets], rebalance_date, signal_cfg)
        alpha = alpha_from_scores(scores, cfg.alpha_scale, cfg.cash_ticker)
        assert_excludes_cash(alpha.index, cfg.cash_ticker, "alpha")
        for asset, score in scores.items():
            signal_records.append(
                {
                    "date": rebalance_date,
                    "asset": asset,
                    "score": float(score),
                    "alpha": float(alpha.loc[asset]),
                }
            )

        standard_market_state = compute_market_state(
            stress_returns_window,
            mode="standard",
            short_window=cfg.stress_short_window_days,
            long_window=cfg.stress_reference_window_days,
            periods_per_year=cfg.periods_per_year,
            risky_tickers=tuple(risky_assets),
            cash_ticker=cfg.cash_ticker,
        )
        ewma_market_state = compute_market_state(
            stress_returns_window,
            mode="ewma",
            ewma_span=cfg.stress_ewma_span_days,
            short_window=cfg.stress_short_window_days,
            long_window=cfg.stress_reference_window_days,
            periods_per_year=cfg.periods_per_year,
            rho_base=cfg.ewma_rho_base,
            rho_lower=cfg.ewma_rho_lower,
            rho_upper=cfg.ewma_rho_upper,
            risky_tickers=tuple(risky_assets),
            cash_ticker=cfg.cash_ticker,
        )

        for spec in strategy_specs:
            pre_full = current_weights[spec.name].reindex(full_assets).fillna(0.0)
            spec_covariance_lookback_days = int(
                spec.params.get("covariance_lookback_days", cfg.covariance_lookback_days)
            )
            spec_covariance_estimator = spec.params.get(
                "covariance_estimator", cfg.covariance_estimator
            )
            spec_covariance_window = history_returns.loc[:, risky_assets].tail(
                spec_covariance_lookback_days
            )
            if len(spec_covariance_window) < spec_covariance_lookback_days:
                continue
            covariance, cov_diag = estimate_covariance_with_diagnostics(
                spec_covariance_window,
                estimator=spec_covariance_estimator,
                periods_per_year=cfg.periods_per_year,
                ewma_decay=spec.params.get("ewma_decay", 0.94),
            )
            assert_excludes_cash(covariance.columns, cfg.cash_ticker, "covariance")
            market_state_mode = _market_state_mode_for_spec(spec)
            market_state = (
                ewma_market_state if market_state_mode == "ewma" else standard_market_state
            )
            rho_t = _rho_for_mode(cfg, market_state.rho_t)
            spec_target_volatility, target_volatility_mode = _target_volatility_for_spec(
                spec, cfg, market_state
            )
            target_full, opt_status, objective, extra_diag = _build_target_weights(
                spec=spec,
                risky_assets=risky_assets,
                alpha=alpha,
                covariance=covariance,
                expected_returns=spec_covariance_window.mean() * cfg.periods_per_year,
                pre_full_weights=pre_full,
                rho_t=rho_t,
                config=cfg,
                effective_target_volatility=spec_target_volatility,
            )
            target_full = target_full.reindex(full_assets).fillna(0.0).astype(float)
            target_full = target_full.clip(lower=0.0)
            target_full = target_full / target_full.sum()
            risky_target = target_full.loc[risky_assets]
            cash_target = float(target_full.loc[cfg.cash_ticker])

            target_vol_for_validation = (
                spec_target_volatility
                if spec.kind == "cash_allowed_target_vol_robust"
                else float("inf")
            )
            validation = validate_cash_allowed_weights(
                risky_weights=risky_target,
                cash_weight=cash_target,
                covariance=covariance,
                upper_bound=spec.params.get("upper_bound", cfg.upper_bound),
                target_volatility=target_vol_for_validation,
                pre_full_weights=pre_full,
                cash_ticker=cfg.cash_ticker,
            )
            rebalance_turnover = one_way_turnover(target_full, pre_full)
            cost = transaction_cost(target_full, pre_full, cfg.transaction_cost_rate)
            net_period_returns, gross_period_returns, ending_weights = simulate_holding_period(
                target_full,
                holding_returns,
                initial_cost=cost,
            )
            daily_net_returns_by_strategy[spec.name].append(net_period_returns)
            daily_gross_returns_by_strategy[spec.name].append(gross_period_returns)
            current_weights[spec.name] = ending_weights.reindex(full_assets).fillna(0.0)

            for asset, weight in target_full.items():
                weights_records.append(
                    {
                        "date": rebalance_date,
                        "strategy": spec.name,
                        "asset": asset,
                        "weight": float(weight),
                    }
                )
            turnover_records.append(
                {"date": rebalance_date, "strategy": spec.name, "turnover": rebalance_turnover}
            )
            diagnostics_records.append(
                {
                    "date": rebalance_date,
                    "strategy": spec.name,
                    "solver_status": opt_status,
                    "solver_success": extra_diag.get("solver_success", opt_status is None),
                    "objective_value": objective,
                    "full_weight_sum": validation.full_weight_sum,
                    "risky_weight_sum": validation.risky_weight_sum,
                    "cash_weight": validation.cash_weight,
                    "min_risky_weight": validation.min_risky_weight,
                    "max_risky_weight": validation.max_risky_weight,
                    "turnover": validation.one_way_turnover,
                    "transaction_cost": cost,
                    "constraint_valid": validation.is_valid,
                    "target_volatility_violation": validation.target_volatility_violation,
                    "predicted_volatility": validation.predicted_volatility,
                    "target_volatility": (
                        spec_target_volatility
                        if spec.kind == "cash_allowed_target_vol_robust"
                        else np.nan
                    ),
                    "covariance_estimator": spec_covariance_estimator,
                    "cov_min_eigenvalue_before": cov_diag.min_eigenvalue_before,
                    "cov_min_eigenvalue_after": cov_diag.min_eigenvalue_after,
                    "cov_psd_repaired": cov_diag.psd_repaired,
                    "alpha_scale": cfg.alpha_scale,
                    "momentum_lookback_days": cfg.momentum_lookback_days,
                    "momentum_skip_days": cfg.momentum_skip_days,
                    "trend_window_days": cfg.trend_window_days,
                    "covariance_lookback_days": spec_covariance_lookback_days,
                    "stress_ewma_span_days": cfg.stress_ewma_span_days,
                    "stress_short_window_days": cfg.stress_short_window_days,
                    "stress_reference_window_days": cfg.stress_reference_window_days,
                    "sigma_ewma_short": market_state.sigma_ewma_short,
                    "sigma_ew_63d": market_state.sigma_ew_63d,
                    "sigma_ew_252d": market_state.sigma_ew_252d,
                    "v_t": market_state.v_t,
                    "rho_t": rho_t,
                    "rho_mode": cfg.rho_mode,
                    "market_state_mode": market_state_mode,
                    "effective_target_volatility": (
                        spec_target_volatility
                        if spec.kind == "cash_allowed_target_vol_robust"
                        else np.nan
                    ),
                    "target_volatility_mode": (
                        target_volatility_mode
                        if spec.kind == "cash_allowed_target_vol_robust"
                        else "n/a"
                    ),
                    "lambda_tc": cfg.lambda_tc,
                    "eta_l2": cfg.eta_l2,
                    "fallback_used": extra_diag.get("fallback_used", False),
                    "robust_penalty": extra_diag.get("robust_penalty"),
                    "solver_time_seconds": extra_diag.get("solver_time_seconds", np.nan),
                    "risk_aversion": extra_diag.get("risk_aversion", np.nan),
                }
            )

    returns_frame = pd.DataFrame(
        {
            name: pd.concat(series_list).sort_index()
            for name, series_list in daily_net_returns_by_strategy.items()
            if series_list
        }
    )
    gross_returns_frame = pd.DataFrame(
        {
            name: pd.concat(series_list).sort_index()
            for name, series_list in daily_gross_returns_by_strategy.items()
            if series_list
        }
    )
    weights_frame = pd.DataFrame(weights_records)
    diagnostics_frame = pd.DataFrame(diagnostics_records)
    turnover_frame = (
        pd.DataFrame(turnover_records)
        .pivot(index="date", columns="strategy", values="turnover")
        .sort_index()
    )
    signals_frame = pd.DataFrame(signal_records)
    performance = performance_summary(
        returns_frame,
        turnover=turnover_frame,
        weights=weights_frame,
        diagnostics=diagnostics_frame,
        cash_ticker=cfg.cash_ticker,
    )
    return BacktestResult(
        returns=returns_frame,
        gross_returns=gross_returns_frame,
        weights=weights_frame,
        diagnostics=diagnostics_frame,
        turnover=turnover_frame,
        signals=signals_frame,
        performance=performance,
    )


def _build_target_weights(
    spec: StrategySpec,
    risky_assets: list[str],
    alpha: pd.Series,
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    pre_full_weights: pd.Series,
    rho_t: float,
    config: BacktestConfig,
    effective_target_volatility: float,
) -> tuple[pd.Series, str | None, float | None, dict[str, Any]]:
    """Dispatch strategy-specific target-weight construction."""
    if spec.kind == "equal_weight":
        return equal_weight_risky(risky_assets, config.cash_ticker), None, None, {"solver_success": True}

    if spec.kind == "minimum_variance":
        result = solve_minimum_variance_long_only(
            covariance=covariance,
            upper_bound=spec.params.get("upper_bound", config.upper_bound),
            solver=spec.params.get("solver"),
        )
        full = full_weight_vector(result.weights.reindex(risky_assets).fillna(0.0), 0.0, config.cash_ticker)
        return full, result.status, result.objective_value, dict(result.diagnostics)

    if spec.kind == "traditional_mvo":
        result = solve_traditional_mvo_long_only(
            expected_returns=expected_returns.reindex(risky_assets).fillna(0.0),
            covariance=covariance,
            risk_aversion=spec.params.get("risk_aversion", 1.0),
            upper_bound=spec.params.get("upper_bound", config.upper_bound),
            solver=spec.params.get("solver"),
        )
        full = full_weight_vector(
            result.weights.reindex(risky_assets).fillna(0.0), 0.0, config.cash_ticker
        )
        return full, result.status, result.objective_value, dict(result.diagnostics)

    if spec.kind == "cash_allowed_target_vol_robust":
        opt_cfg = CashAllowedOptimizationConfig(
            target_volatility=effective_target_volatility,
            lambda_tc=spec.params.get("lambda_tc", config.lambda_tc),
            eta_l2=spec.params.get("eta_l2", config.eta_l2),
            upper_bound=spec.params.get("upper_bound", config.upper_bound),
            solver=spec.params.get("solver"),
        )
        assert_fixed_lambda_tc(opt_cfg.lambda_tc)
        result = solve_cash_allowed_target_vol_robust(
            alpha=alpha.reindex(risky_assets),
            covariance=covariance.reindex(index=risky_assets, columns=risky_assets),
            pre_risky_weights=pre_full_weights.reindex(risky_assets).fillna(0.0),
            pre_cash_weight=float(pre_full_weights.get(config.cash_ticker, 0.0)),
            rho_t=rho_t,
            config=opt_cfg,
            cash_ticker=config.cash_ticker,
        )
        return result.full_weights, result.status, result.objective_value, dict(result.diagnostics)

    raise ValueError(f"Unsupported strategy kind: {spec.kind}")


def _target_volatility_for_spec(
    spec: StrategySpec,
    config: BacktestConfig,
    market_state,
) -> tuple[float, str]:
    """Return the target volatility and target-vol mode used by this strategy."""
    mode: TargetVolatilityMode = spec.params.get("target_volatility_mode", "fixed")
    if mode == "fixed":
        return float(spec.params.get("target_volatility", config.target_volatility)), mode
    if mode == "inverse_vt":
        base = float(spec.params.get("target_volatility_base", config.target_volatility))
        lower = float(spec.params.get("target_volatility_lower", 0.04))
        upper = float(spec.params.get("target_volatility_upper", 0.16))
        epsilon = float(spec.params.get("target_volatility_epsilon", 1e-12))
        target = np.clip(base / max(float(market_state.v_t), epsilon), lower, upper)
        return float(target), mode
    raise ValueError(f"Unsupported target_volatility_mode: {mode}")


def _market_state_mode_for_spec(spec: StrategySpec) -> str:
    """Return the market-state mode used by this strategy specification."""
    mode: MarketStateMode = spec.params.get("market_state_mode", "standard")
    if mode not in {"standard", "ewma"}:
        raise ValueError(f"Unsupported market_state_mode: {mode}")
    return mode


def _rho_for_mode(config: BacktestConfig, dynamic_rho: float) -> float:
    if config.rho_mode == "dynamic":
        return dynamic_rho
    if config.rho_mode == "zero":
        return 0.0
    if config.rho_mode == "fixed":
        return config.fixed_rho
    raise ValueError(f"Unknown rho_mode: {config.rho_mode}")

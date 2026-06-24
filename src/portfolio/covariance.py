"""Covariance estimators and PSD utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
import pandas as pd

from .data import CASH_TICKER, RISKY_TICKERS

CovarianceEstimator = Literal["sample", "ledoit_wolf", "ewma"]


@dataclass(frozen=True)
class CovarianceDiagnostics:
    """Diagnostics for PSD repair."""

    min_eigenvalue_before: float
    min_eigenvalue_after: float
    psd_repaired: bool


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    cleaned = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if cleaned.empty:
        raise ValueError("No valid returns are available for covariance estimation.")
    return cleaned


def risky_returns_frame(
    returns: pd.DataFrame,
    risky_tickers: list[str] | tuple[str, ...] = tuple(RISKY_TICKERS),
    cash_ticker: str = CASH_TICKER,
) -> pd.DataFrame:
    """Return risky ETF returns only, explicitly excluding cash."""
    columns = [ticker for ticker in risky_tickers if ticker in returns.columns and ticker != cash_ticker]
    if not columns:
        raise ValueError("No risky return columns are available.")
    return returns.loc[:, columns]


def sample_covariance(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Estimate annualized sample covariance."""
    cleaned = _clean_returns(returns)
    return cleaned.cov() * periods_per_year


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Estimate annualized Ledoit-Wolf shrinkage covariance."""
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for Ledoit-Wolf covariance. "
            "Install requirements.txt first."
        ) from exc

    cleaned = _clean_returns(returns)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model = LedoitWolf().fit(cleaned.values)
    covariance = model.covariance_ * periods_per_year
    if not np.isfinite(covariance).all():
        raise ValueError("Ledoit-Wolf covariance contains non-finite values.")
    return pd.DataFrame(covariance, index=cleaned.columns, columns=cleaned.columns)


def ewma_covariance(
    returns: pd.DataFrame,
    decay: float = 0.94,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Estimate annualized EWMA covariance using exponentially decaying weights."""
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be between 0 and 1.")
    cleaned = _clean_returns(returns)
    demeaned = cleaned - cleaned.mean(axis=0)
    n_obs = len(demeaned)
    raw_weights = np.array([(1.0 - decay) * decay ** k for k in range(n_obs - 1, -1, -1)])
    weights = raw_weights / raw_weights.sum()
    weighted = demeaned.values * np.sqrt(weights[:, None])
    covariance = weighted.T @ weighted * periods_per_year
    return pd.DataFrame(covariance, index=cleaned.columns, columns=cleaned.columns)


def estimate_covariance(
    returns: pd.DataFrame,
    estimator: CovarianceEstimator = "ledoit_wolf",
    periods_per_year: int = 252,
    ewma_decay: float = 0.94,
    psd_floor: float = 1e-8,
) -> pd.DataFrame:
    """Estimate and PSD-repair an annualized covariance matrix."""
    covariance, _ = estimate_covariance_with_diagnostics(
        returns=returns,
        estimator=estimator,
        periods_per_year=periods_per_year,
        ewma_decay=ewma_decay,
        psd_floor=psd_floor,
    )
    return covariance


def estimate_covariance_with_diagnostics(
    returns: pd.DataFrame,
    estimator: CovarianceEstimator = "ledoit_wolf",
    periods_per_year: int = 252,
    ewma_decay: float = 0.94,
    psd_floor: float = 1e-8,
) -> tuple[pd.DataFrame, CovarianceDiagnostics]:
    """Estimate annualized covariance and report PSD repair diagnostics."""
    if estimator == "sample":
        covariance = sample_covariance(returns, periods_per_year)
    elif estimator == "ledoit_wolf":
        covariance = ledoit_wolf_covariance(returns, periods_per_year)
    elif estimator == "ewma":
        covariance = ewma_covariance(returns, ewma_decay, periods_per_year)
    else:
        raise ValueError(f"Unknown covariance estimator: {estimator}")
    return make_psd_with_diagnostics(covariance, floor=psd_floor)


def make_psd(covariance: pd.DataFrame, floor: float = 1e-8) -> pd.DataFrame:
    """Symmetrize and eigenvalue-clip a covariance matrix."""
    repaired, _ = make_psd_with_diagnostics(covariance, floor=floor)
    return repaired


def make_psd_with_diagnostics(
    covariance: pd.DataFrame,
    floor: float = 1e-8,
) -> tuple[pd.DataFrame, CovarianceDiagnostics]:
    """Symmetrize, eigenvalue-clip, and report PSD repair diagnostics."""
    values = covariance.values.astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Covariance matrix contains non-finite values.")
    symmetric = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, floor)
    repaired = np.einsum("ik,k,jk->ij", eigenvectors, clipped, eigenvectors, optimize=False)
    repaired = 0.5 * (repaired + repaired.T)
    repaired_values = np.linalg.eigvalsh(repaired)
    diagnostics = CovarianceDiagnostics(
        min_eigenvalue_before=float(eigenvalues.min()),
        min_eigenvalue_after=float(repaired_values.min()),
        psd_repaired=bool(eigenvalues.min() < -floor or eigenvalues.min() < floor),
    )
    return pd.DataFrame(repaired, index=covariance.index, columns=covariance.columns), diagnostics


def is_psd(covariance: pd.DataFrame, tolerance: float = 1e-8) -> bool:
    """Return True when the covariance matrix is positive semidefinite."""
    values = 0.5 * (covariance.values + covariance.values.T)
    min_eigenvalue = np.linalg.eigvalsh(values).min()
    return bool(min_eigenvalue >= -tolerance)


def covariance_sqrt(covariance: pd.DataFrame, floor: float = 1e-10) -> np.ndarray:
    """Return a matrix factor F such that F.T @ F approximates covariance."""
    values = 0.5 * (covariance.values + covariance.values.T)
    if not np.isfinite(values).all():
        raise ValueError("Covariance matrix contains non-finite values.")
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    clipped = np.maximum(eigenvalues, floor)
    return np.sqrt(clipped)[:, None] * eigenvectors.T

import importlib.util
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")


def load_report_assets_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "05_generate_report_assets.py"
    spec = importlib.util.spec_from_file_location("generate_report_assets", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cumulative_plot_returns_include_baselines_and_final_strategy_only():
    module = load_report_assets_module()
    dates = pd.bdate_range("2021-01-01", periods=5)
    returns = pd.DataFrame(
        {
            "Equal Weight": [0.002, 0.003, 0.004, 0.004, 0.005],
            "Minimum Variance": [0.001, 0.002, 0.002, 0.003, 0.003],
            "Traditional MVO": [0.002, 0.002, 0.004, 0.003, 0.004],
            "Proposed Strategy": [0.003, 0.004, 0.005, 0.006, 0.007],
            "Extra Strategy": [0.001, 0.001, 0.001, 0.001, 0.001],
        },
        index=dates,
    )
    combined = module._build_cumulative_plot_returns(returns)

    assert list(combined.columns) == [
        "Equal Weight",
        "Minimum Variance",
        "Traditional MVO",
        "Proposed Strategy",
    ]
    assert combined.index.min() == dates[0]

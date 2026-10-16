"""
SMC Stochastic Volatility Package (`src`)

Exports the base state-space model, dataset loaders, evaluation metrics,
and parallel runner utilities for all four project roles.
"""

from .base_model import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    PLOTS_DIR,
    PROJECT_ROOT,
    SIMULATED_CSV_PATH,
    SP500_CLEANED_CSV_PATH,
    SP500_FULL_CLEANED_CSV_PATH,
    SP500_RAW_CSV_PATH,
    TRUE_SIM_PARAMS,
    StochasticVolatility,
    StochVol,
    evaluate_filter_performance,
    generate_simulated_dataset,
    load_simulated_dataset,
    load_sp500_dataset,
    plot_sandbox_validation,
    plot_sp500_overview_and_baseline,
    prepare_sp500_dataset,
    run_baseline_bootstrap,
    run_parallel_smc,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_RAW_DIR",
    "DATA_PROCESSED_DIR",
    "PLOTS_DIR",
    "SIMULATED_CSV_PATH",
    "SP500_RAW_CSV_PATH",
    "SP500_CLEANED_CSV_PATH",
    "SP500_FULL_CLEANED_CSV_PATH",
    "TRUE_SIM_PARAMS",
    "StochVol",
    "StochasticVolatility",
    "generate_simulated_dataset",
    "load_simulated_dataset",
    "prepare_sp500_dataset",
    "load_sp500_dataset",
    "evaluate_filter_performance",
    "run_baseline_bootstrap",
    "run_parallel_smc",
    "plot_sandbox_validation",
    "plot_sp500_overview_and_baseline",
]

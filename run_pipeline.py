#!/usr/bin/env python3
"""
Parallelized Runner Script (`run_pipeline.py`) — Data & Base Setup (Gabriel)

Executes the Data & Base Setup pipeline across Milestones 1, 2, and 3:
- M1: Instantiates the base `StochVol` StateSpaceModel class in `particles`
- M2: Generates/loads the synthetic sandbox dataset (`data/processed/simulated_sv_data.csv`)
      and verifies that the baseline Bootstrap particle filter runs cleanly
- M3: Cleans/loads the S&P 500 log-returns CSV (`data/processed/sp500_log_returns.csv`)
      and executes parallelized SMC runs across CPU cores via `particles.multiSMC`

Usage:
    python run_pipeline.py --particles 1000 --nruns 8 --nprocs 0
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib.pyplot as plt

from src.base_model import (
    SIMULATED_CSV_PATH,
    SP500_CLEANED_CSV_PATH,
    TRUE_SIM_PARAMS,
    StochVol,
    generate_simulated_dataset,
    load_simulated_dataset,
    load_sp500_dataset,
    plot_sandbox_validation,
    plot_sp500_overview_and_baseline,
    prepare_sp500_dataset,
    run_baseline_bootstrap,
    run_parallel_smc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the parallelized Data & Base Setup SMC pipeline."
    )
    parser.add_argument(
        "--particles",
        "-N",
        type=int,
        default=1000,
        help="Number of particles per SMC run (default: 1000).",
    )
    parser.add_argument(
        "--nruns",
        type=int,
        default=8,
        help="Number of independent parallel runs in multiSMC (default: 8).",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=0,
        help="Number of CPU cores for multiSMC (0 = all available cores).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    print("=" * 76)
    print("SMC STOCHASTIC VOLATILITY — DATA & BASE SETUP PARALLEL RUNNER")
    print("=" * 76)

    # 1. Synthetic Sandbox Data (M1 & M2)
    generate_simulated_dataset(save_csv=True)
    sim_data = load_simulated_dataset()
    print(
        f"[M1/M2] Synthetic Sandbox Data : {SIMULATED_CSV_PATH.name} "
        f"(T={len(sim_data['y_obs'])}, mu={TRUE_SIM_PARAMS['mu']}, "
        f"phi={TRUE_SIM_PARAMS['phi']}, sigma={TRUE_SIM_PARAMS['sigma']})"
    )

    # 2. Baseline Bootstrap PF + Parallel multiSMC Verification on Sandbox Data (M2)
    boot_sim = run_baseline_bootstrap(
        y_data=sim_data["y_obs"],
        model=StochVol(**TRUE_SIM_PARAMS),
        N=args.particles,
        x_true=sim_data["x_true"],
    )
    par_sim = run_parallel_smc(
        y_data=sim_data["y_obs"],
        model=StochVol(**TRUE_SIM_PARAMS),
        N=args.particles,
        nruns=args.nruns,
        nprocs=args.nprocs,
        x_true=sim_data["x_true"],
    )
    print(
        f"[M2]    Sandbox Bootstrap PF   : RMSE(X) = {boot_sim['metrics']['rmse_x']:.4f}, "
        f"MSE(X) = {boot_sim['metrics']['mse_x']:.4f}, "
        f"Mean ESS = {boot_sim['metrics']['mean_ess']:.1f}/{args.particles}"
    )
    print(
        f"[M2/M3] Parallel multiSMC ({par_sim['nruns']}x) : "
        f"RMSE(X) = {par_sim['rmse_mean']:.4f} +/- {par_sim['rmse_std']:.4f}, "
        f"logL = {par_sim['logL_mean']:.2f} +/- {par_sim['logL_std']:.2f}"
    )

    fig_sim = plot_sandbox_validation(sim_data, boot_sim)
    plt.close(fig_sim)

    # 3. Cleaned S&P 500 Dataset & Parallel Runner on Real Market Data (M3)
    prepare_sp500_dataset()
    sp500_data = load_sp500_dataset()
    sp_df = sp500_data["df"]
    print(
        f"[M3]    Cleaned S&P 500 Data   : {SP500_CLEANED_CSV_PATH.name} "
        f"(T={len(sp500_data['y_obs'])}, "
        f"{sp_df['Date'].iloc[0].strftime('%Y-%m-%d')} to {sp_df['Date'].iloc[-1].strftime('%Y-%m-%d')})"
    )

    sp500_model = StochVol(mu=-0.8, rho=0.97, sigma=0.25)
    boot_sp500 = run_baseline_bootstrap(
        y_data=sp500_data["y_obs"],
        model=sp500_model,
        N=args.particles,
    )
    par_sp500 = run_parallel_smc(
        y_data=sp500_data["y_obs"],
        model=sp500_model,
        N=args.particles,
        nruns=args.nruns,
        nprocs=args.nprocs,
    )
    print(
        f"[M3]    S&P 500 Parallel ({par_sp500['nruns']}x) : "
        f"Mean ESS = {par_sp500['mean_ess']:.1f}/{args.particles}, "
        f"logL = {par_sp500['logL_mean']:.2f} +/- {par_sp500['logL_std']:.2f}"
    )

    fig_sp = plot_sp500_overview_and_baseline(sp500_data, boot_sp500)
    plt.close(fig_sp)
    print("[M2/M3] Saved plots to `plots/m2_sandbox_bootstrap_validation.png` and `plots/m3_sp500_overview_and_baseline.png`")

    elapsed = time.perf_counter() - t0
    print("=" * 76)
    print(f"Data & Base Setup pipeline completed in {elapsed:.2f}s.")
    print("=" * 76)


if __name__ == "__main__":
    main()

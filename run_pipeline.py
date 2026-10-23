#!/usr/bin/env python3
"""
Data & Base ChMSV Model Setup Runner (`run_pipeline.py`) — Role 1 (Gabriel)

Executes Role 1 deliverables based on Szczepocki (2023):
1. Cleans and loads the real 4-asset dataset (SPX, Gold, Bitcoin, Ethereum,
   4 June 2018 – 4 June 2023, T = 1258 daily percentage log-returns) and
   verifies Table 1 descriptive statistics.
2. Generates and loads the simulated test datasets:
   - Fast 2-asset version (SPX & Gold, p = 2, 3 states, 9 parameters, T = 500)
   - Full 4-asset version (SPX, Gold, BTC, ETH, p = 4, 10 states, 30 parameters, T = 1258)
3. Verifies the `ChMSV` (`ChMSV4Asset`, `ChMSV2Asset`) StateSpaceModel classes in
   `particles` and the state-to-volatility/correlation transformations (Equations 4–7).
4. Saves Role 1 data overview figures to `plots/`.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib.pyplot as plt
from particles import state_space_models as ssm

from src.base_model import (
    PROJECT_ROOT,
    REAL_2ASSETS_CSV_PATH,
    REAL_4ASSETS_CSV_PATH,
    SIM_2ASSETS_CSV_PATH,
    SIM_4ASSETS_CSV_PATH,
    ChMSV2Asset,
    ChMSV4Asset,
    compute_table1_summary_stats,
    generate_simulated_chmsv_dataset,
    load_real_dataset,
    load_simulated_chmsv_dataset,
    plot_figure1_real_data,
    plot_simulated_datasets_overview,
    prepare_real_4assets_dataset,
)


def main() -> None:
    t0 = time.perf_counter()

    print("=" * 78)
    print("SMC ChMSV PROJECT — ROLE 1: DATA & BASE ChMSV MODEL SETUP (Szczepocki, 2023)")
    print("=" * 78)

    # 1. Real 4-Asset & 2-Asset Datasets (June 4, 2018 – June 4, 2023)
    real_df = prepare_real_4assets_dataset()
    real4 = load_real_dataset(p=4)
    real2 = load_real_dataset(p=2)
    print(
        f"\n[M1] Real 4-Asset Dataset : {REAL_4ASSETS_CSV_PATH.relative_to(PROJECT_ROOT)} "
        f"(shape: {real4['y_obs'].shape}, {real_df['Date'].iloc[0]} to {real_df['Date'].iloc[-1]})"
    )
    print(
        f"[M1] Real 2-Asset Subset  : {REAL_2ASSETS_CSV_PATH.relative_to(PROJECT_ROOT)} "
        f"(shape: {real2['y_obs'].shape})"
    )
    print("\nTable 1 Descriptive Statistics (100 * daily log-returns, T = 1258):")
    print(compute_table1_summary_stats(real_df).to_string(index=False))

    # 2. Simulated Test Datasets (2-Asset Fast & 4-Asset Full using Table 2 Parameters)
    generate_simulated_chmsv_dataset(p=2, T=500, seed=42, save_csv=True)
    generate_simulated_chmsv_dataset(p=4, T=1258, seed=42, save_csv=True)
    sim2 = load_simulated_chmsv_dataset(p=2)
    sim4 = load_simulated_chmsv_dataset(p=4)

    print(
        f"\n[M1] Simulated 2-Asset Dataset : {SIM_2ASSETS_CSV_PATH.relative_to(PROJECT_ROOT)} "
        f"(T={sim2['T']}, p={sim2['p']}, d_x={sim2['dx']} states, {3 * sim2['dx']} parameters)"
    )
    print(
        f"[M1] Simulated 4-Asset Dataset : {SIM_4ASSETS_CSV_PATH.relative_to(PROJECT_ROOT)} "
        f"(T={sim4['T']}, p={sim4['p']}, d_x={sim4['dx']} states, {3 * sim4['dx']} parameters)"
    )

    # 3. Verify Base ChMSV StateSpaceModel & State-to-Correlation Helpers (M1 & M2)
    for label, model, data_arr in [
        ("ChMSV2Asset (p=2, 3 states, 9 params)", ChMSV2Asset(), sim2["y_obs"]),
        ("ChMSV4Asset (p=4, 10 states, 30 params)", ChMSV4Asset(), sim4["y_obs"]),
    ]:
        fk = ssm.Bootstrap(ssm=model, data=data_arr)
        x0 = fk.M0(32)
        x1 = fk.M(1, x0)
        log_weights = fk.logG(0, x0, x1)
        transformed = model.transform_states(x1)
        print(
            f"[M1/M2] Verified {label}: "
            f"X_t shape={x1.shape}, logG shape={log_weights.shape}, "
            f"stdev shape={transformed['stdev'].shape}, "
            f"corr_pairs shape={transformed['corr_pairs'].shape}"
        )

    # 4. Generate Role 1 Data Overview Plots
    fig1 = plot_figure1_real_data(real4)
    plt.close(fig1)
    fig_sim = plot_simulated_datasets_overview(sim4, sim2)
    plt.close(fig_sim)
    print(
        "\n[M1/M2] Saved data overview plots:\n"
        "  * plots/fig1_real_4assets_prices_and_returns.png\n"
        "  * plots/simulated_chmsv_datasets_overview.png"
    )

    elapsed = time.perf_counter() - t0
    print("=" * 78)
    print(f"Role 1 (Data & Base ChMSV Model Setup) completed cleanly in {elapsed:.2f}s.")
    print("=" * 78)


if __name__ == "__main__":
    main()

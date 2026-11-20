"""
Bootstrap Particle Filter & Dynamic Correlation Tracking

Deliverables:
- `run_bootstrap_filter`: Runs SMC with N particles and extracts daily volatilities, dynamic correlations, log-likelihood, and ESS.
- `run_10run_variability_benchmark`: Evaluates Monte Carlo standard errors over 10 independent runs (Table 2 check).
- `plot_figure2_volatilities`: Reproduction of Figure 2 (Daily standard deviations).
- `plot_figure3_spx_correlations`: Reproduction of Figure 3 (Dynamic correlations with SPX).
- `plot_figure4_candidate_correlations`: Reproduction of Figure 4 (Cross-correlations between safe havens).
- `plot_simulation_tracking_benchmark`: Compares estimated volatilities and correlations against their true values on simulated data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import particles
from particles import state_space_models as ssm

from src.base_model import (
    ASSET_NAMES_4,
    PLOTS_DIR,
    PROJECT_ROOT,
    TABLE2_LOG_LIKELIHOOD,
    TABLE2_LOG_LIKELIHOOD_SE,
    TABLE2_PARAMS_4ASSET,
    ChMSV,
    ChMSV2Asset,
    ChMSV4Asset,
    get_offdiag_pairs,
    load_real_dataset,
    load_simulated_chmsv_dataset,
    states_to_volatility_and_correlation,
)


def _format_save_path(save_path: Path) -> str:
    """Format a path relative to PROJECT_ROOT safely (works from root or notebooks/)."""
    try:
        return save_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return save_path.as_posix()


# ---------------------------------------------------------------------------
# 1. Core Particle Filtering Routine
# ---------------------------------------------------------------------------
def run_bootstrap_filter(
    y_obs: np.ndarray,
    model: Optional[ChMSV] = None,
    p: int = 4,
    N: int = 1000,
    ESSrmin: float = 0.5,
    resampling: str = "systematic",
    seed: Optional[int] = None,
    compute_quantiles: bool = True,
) -> Dict[str, Any]:
    r"""Execute a Bootstrap Particle Filter on multivariate return series y_obs.

    Uses `particles.SMC` with the Feynman-Kac Bootstrap model:
        M_t = P_t(x_{t-1}, dx_t),   G_t = f_t(y_t | x_t)
    and adaptive resampling triggered when ESS(W_{t-1}^{1:N}) < ESSrmin * N.

    Parameters
    ----------
    y_obs : np.ndarray
        Array of observed returns of shape `(T, p)`.
    model : ChMSV, optional
        Instantiated ChMSV model. If None, defaults to `ChMSV4Asset()` or `ChMSV2Asset()`.
    p : int
        Number of assets (default: 4).
    N : int
        Number of particles (default: 1000 as in Szczepocki, 2023).
    ESSrmin : float
        Adaptive resampling threshold \gamma in [0, 1] (default: 0.5; use 1.0 to resample every step).
    resampling : str
        Resampling algorithm: `'systematic'` (default, minimal variance) or `'multinomial'`.
    seed : int, optional
        Random seed for reproducibility.
    compute_quantiles : bool
        If True, computes 5th and 95th percentiles (90% Bayesian credibility intervals).

    Returns
    -------
    dict
        Dictionary containing:
        - `'log_likelihood'`: Total log-likelihood \log \hat{L}_T^N
        - `'ess'`: ESS trajectory of shape `(T,)`
        - `'x_filtered'`: Posterior mean of latent states \hat{X}_t of shape `(T, d_x)`
        - `'stdev_filtered'`: Filtered asset volatilities (Eq. 6) of shape `(T, p)`
        - `'corr_filtered'`: Filtered dynamic correlations (Eq. 7) of shape `(T, p*(p-1)//2)`
        - `'stdev_q05'`, `'stdev_q95'`: Credibility bands for volatilities
        - `'corr_q05'`, `'corr_q95'`: Credibility bands for dynamic correlations
    """
    if seed is not None:
        np.random.seed(seed)

    y_arr = np.asarray(y_obs, dtype=float)
    if model is None:
        model = ChMSV4Asset() if p == 4 else ChMSV2Asset()

    T = len(y_arr)
    dx = model.dx
    n_pairs = p * (p - 1) // 2

    # Setup Feynman-Kac Bootstrap model in `particles`
    fk = ssm.Bootstrap(ssm=model, data=y_arr)

    # Run Sequential Monte Carlo with adaptive ESS resampling
    # store_history=True keeps full particle history to compute expectations and quantiles
    alg = particles.SMC(
        fk=fk,
        N=N,
        ESSrmin=ESSrmin,
        resampling=resampling,
        store_history=True,
        verbose=False,
    )
    alg.run()

    # Preallocate output arrays
    x_filtered = np.zeros((T, dx), dtype=float)
    stdev_filtered = np.zeros((T, p), dtype=float)
    corr_filtered = np.zeros((T, n_pairs), dtype=float)
    ess_history = np.zeros(T, dtype=float)

    stdev_q05 = np.zeros((T, p), dtype=float) if compute_quantiles else None
    stdev_q95 = np.zeros((T, p), dtype=float) if compute_quantiles else None
    corr_q05 = np.zeros((T, n_pairs), dtype=float) if compute_quantiles else None
    corr_q95 = np.zeros((T, n_pairs), dtype=float) if compute_quantiles else None

    # Process filtered particles step-by-step
    for t in range(T):
        # Normalized importance weights W_t^n such that sum(W_t^n) = 1
        W = alg.hist.wgts[t].W  # shape (N,)
        X = alg.hist.X[t]       # shape (N, dx)

        # 1. Effective Sample Size (Course Section: ESS = 1 / sum(W^2))
        ess_history[t] = 1.0 / np.sum(W**2)

        # 2. Filtered state expectation: E[X_t | y_{0:t}] \approx \sum W_t^n X_t^n
        x_filtered[t] = np.sum(W[:, None] * X, axis=0)

        # 3. Transform each particle into its volatilities (Eq. 6) & correlations (Eq. 7)
        # Vectorized call across all N particles: shape (N, p) and (N, n_pairs)
        trans = states_to_volatility_and_correlation(X, p=p)
        part_sd = trans["stdev"]         # shape (N, p)
        part_corr = trans["corr_pairs"]  # shape (N, n_pairs)

        # Expected volatilities and correlations under empirical measure Q_t^N
        stdev_filtered[t] = np.sum(W[:, None] * part_sd, axis=0)
        corr_filtered[t] = np.sum(W[:, None] * part_corr, axis=0)

        # 4. Weighted quantiles (90% credibility intervals, safeguarded against float rounding)
        if compute_quantiles:
            idx_sort = np.argsort(part_sd, axis=0)
            for j in range(p):
                order = idx_sort[:, j]
                cum_w = np.cumsum(W[order])
                cum_w /= cum_w[-1]
                i05 = min(int(np.searchsorted(cum_w, 0.05)), N - 1)
                i95 = min(int(np.searchsorted(cum_w, 0.95)), N - 1)
                stdev_q05[t, j] = part_sd[order[i05], j]
                stdev_q95[t, j] = part_sd[order[i95], j]

            idx_corr = np.argsort(part_corr, axis=0)
            for k in range(n_pairs):
                order = idx_corr[:, k]
                cum_w = np.cumsum(W[order])
                cum_w /= cum_w[-1]
                i05 = min(int(np.searchsorted(cum_w, 0.05)), N - 1)
                i95 = min(int(np.searchsorted(cum_w, 0.95)), N - 1)
                corr_q05[t, k] = part_corr[order[i05], k]
                corr_q95[t, k] = part_corr[order[i95], k]

    return {
        "log_likelihood": float(alg.logLt),
        "ess": ess_history,
        "x_filtered": x_filtered,
        "stdev_filtered": stdev_filtered,
        "corr_filtered": corr_filtered,
        "stdev_q05": stdev_q05,
        "stdev_q95": stdev_q95,
        "corr_q05": corr_q05,
        "corr_q95": corr_q95,
        "T": T,
        "N": N,
        "p": p,
        "ESSrmin": ESSrmin,
    }


# ---------------------------------------------------------------------------
# 2. Monte Carlo Variability Benchmark (Table 2 & Milestone 2)
# ---------------------------------------------------------------------------
def run_10run_variability_benchmark(
    y_obs: np.ndarray,
    model: Optional[ChMSV] = None,
    p: int = 4,
    N: int = 1000,
    ESSrmin: float = 0.5,
    n_runs: int = 10,
    base_seed: int = 100,
) -> Dict[str, Any]:
    """Execute multiple independent runs of the Particle Filter to evaluate Monte Carlo variability.

    Szczepocki (2023, Table 2, p. 52) obtained:
        log-likelihood = -9972.7344 with standard error = 0.6468 (averaged across 12 runs with N = 1000).

    This benchmark reproduces this assessment and measures the stability of the SMC estimator.
    """
    print(f"\n[Benchmark] Running {n_runs} independent SMC runs (N = {N}, p = {p}, ESSrmin = {ESSrmin})...")
    log_liks: List[float] = []
    mean_ess_list: List[float] = []
    min_ess_list: List[float] = []

    for r in range(n_runs):
        seed = base_seed + r
        res = run_bootstrap_filter(
            y_obs=y_obs,
            model=model,
            p=p,
            N=N,
            ESSrmin=ESSrmin,
            resampling="systematic",
            seed=seed,
            compute_quantiles=False,
        )
        ll = res["log_likelihood"]
        mean_ess = float(np.mean(res["ess"]))
        min_ess = float(np.min(res["ess"]))

        log_liks.append(ll)
        mean_ess_list.append(mean_ess)
        min_ess_list.append(min_ess)
        print(
            f"   Run {r+1:02d}/{n_runs:02d} (seed={seed}): "
            f"Log-Likelihood = {ll:10.4f} | Mean ESS = {mean_ess:.1f}"
        )

    log_liks_arr = np.array(log_liks, dtype=float)
    mean_ll = float(np.mean(log_liks_arr))
    std_ll = float(np.std(log_liks_arr, ddof=1)) if n_runs > 1 else 0.0
    se_ll = std_ll / np.sqrt(n_runs)

    # pomp::logmeanexp-style log-likelihood aggregation and jackknife SE
    m = float(np.max(log_liks_arr))
    lme_ll = float(m + np.log(np.mean(np.exp(log_liks_arr - m))))
    if n_runs > 1:
        jk = np.array([
            m + np.log(np.mean(np.exp(np.delete(log_liks_arr, i) - m)))
            for i in range(n_runs)
        ])
        lme_se = float(np.sqrt(((n_runs - 1) / n_runs) * np.sum((jk - np.mean(jk)) ** 2)))
    else:
        lme_se = 0.0

    print("-" * 65)
    print(f"   SMC Log-Likelihood Mean: {mean_ll:.4f} (Std: {std_ll:.4f}, SE: {se_ll:.4f})")
    print(f"   Log-Mean-Exp (pomp): {lme_ll:.4f} (Jackknife SE: {lme_se:.4f})")
    if p == 4:
        print(f"   Paper Table 2 Value   : {TABLE2_LOG_LIKELIHOOD:.4f} (SE: {TABLE2_LOG_LIKELIHOOD_SE:.4f})")
        diff = abs(mean_ll - TABLE2_LOG_LIKELIHOOD)
        print(f"   Absolute Difference    : {diff:.4f}")
    print("-" * 65)

    return {
        "log_liks": log_liks_arr,
        "mean_log_lik": mean_ll,
        "std_log_lik": std_ll,
        "se_log_lik": se_ll,
        "logmeanexp_log_lik": lme_ll,
        "logmeanexp_se": lme_se,
        "mean_ess": float(np.mean(mean_ess_list)),
        "min_ess": float(np.mean(min_ess_list)),
        "n_runs": n_runs,
        "N": N,
        "p": p,
    }


# ---------------------------------------------------------------------------
# 3. Figure 2 Reproduction: Filtered Volatilities (Szczepocki 2023, p. 53)
# ---------------------------------------------------------------------------
def plot_figure2_volatilities(
    dates: pd.Series,
    stdev_filtered: np.ndarray,
    stdev_q05: Optional[np.ndarray] = None,
    stdev_q95: Optional[np.ndarray] = None,
    save_path: Optional[Path] = PLOTS_DIR / "fig2_reproduction_volatilities.png",
) -> plt.Figure:
    """Reproduce Figure 2 (Szczepocki, 2023, p. 53): Estimates of the standard deviation of returns.

    Displays the time-varying volatility sqrt(Sigma_{t,ii}) for SPX, Gold, Bitcoin, and Ethereum.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    asset_names = ASSET_NAMES_4
    colors = ["#1F4E79", "#B8860B", "#D95F02", "#5E3C99"]

    for i in range(4):
        ax = axes[i]
        ax.plot(
            dates,
            stdev_filtered[:, i],
            color=colors[i],
            lw=1.1,
            label=f"{asset_names[i]} Filtered Stdev (Eq. 6)",
        )

        if stdev_q05 is not None and stdev_q95 is not None:
            ax.fill_between(
                dates,
                stdev_q05[:, i],
                stdev_q95[:, i],
                color=colors[i],
                alpha=0.20,
                label="90% Particle Credibility Band",
            )

        ax.set_ylabel(f"{asset_names[i]} Stdev", fontsize=9.5, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8.5)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date", fontsize=10, fontweight="bold")

    fig.suptitle(
        "Reproduction of Figure 2 (Szczepocki, 2023): Estimates of Return Standard Deviation\n"
        "(Bootstrap Particle Filter with N = 1000 particles, Table 2 Parameters)",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[Plot] Saved Figure 2 to: {_format_save_path(save_path)}")

    return fig


# ---------------------------------------------------------------------------
# 4. Figure 3 Reproduction: Dynamic Correlations with SPX (Szczepocki 2023, p. 53)
# ---------------------------------------------------------------------------
def plot_figure3_spx_correlations(
    dates: pd.Series,
    corr_filtered: np.ndarray,
    corr_q05: Optional[np.ndarray] = None,
    corr_q95: Optional[np.ndarray] = None,
    save_path: Optional[Path] = PLOTS_DIR / "fig3_reproduction_spx_correlations.png",
) -> plt.Figure:
    """Reproduce Figure 3 (Szczepocki, 2023, p. 53): Dynamic correlations between SPX and safe havens.

    Pairs plotted:
    - Pair (1,2): SPX and Gold
    - Pair (1,3): SPX and Bitcoin
    - Pair (1,4): SPX and Ethereum
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    pair_indices = [0, 1, 2]  # Indices for pairs (1,2), (1,3), (1,4) in Table 2 ordering
    pair_titles = [
        "Dynamic Correlation: SPX and Gold (Pair 1-2)",
        "Dynamic Correlation: SPX and Bitcoin (Pair 1-3)",
        "Dynamic Correlation: SPX and Ethereum (Pair 1-4)",
    ]
    colors = ["#B8860B", "#D95F02", "#5E3C99"]

    for idx, (pair_idx, title, color) in enumerate(zip(pair_indices, pair_titles, colors)):
        ax = axes[idx]
        ax.plot(
            dates,
            corr_filtered[:, pair_idx],
            color=color,
            lw=1.2,
            label=r"Estimated $\rho_{1j,t}$ (Eq. 7)",
        )

        if corr_q05 is not None and corr_q95 is not None:
            ax.fill_between(
                dates,
                corr_q05[:, pair_idx],
                corr_q95[:, pair_idx],
                color=color,
                alpha=0.20,
                label="90% Particle Credibility Band",
            )

        ax.axhline(0.0, color="black", linestyle="--", lw=0.8, alpha=0.7)
        ax.set_ylabel(r"Correlation $\rho_t$", fontsize=9.5, fontweight="bold")
        ax.set_title(title, fontsize=10, fontweight="bold")
        low_series = corr_q05[:, pair_idx] if corr_q05 is not None else corr_filtered[:, pair_idx]
        high_series = corr_q95[:, pair_idx] if corr_q95 is not None else corr_filtered[:, pair_idx]
        y_lo = float(np.min(low_series))
        y_hi = float(np.max(high_series))
        pad = max(0.02, 0.12 * (y_hi - y_lo))
        ax.set_ylim(min(-0.02, y_lo - pad), y_hi + pad)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8.5)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date", fontsize=10, fontweight="bold")

    fig.suptitle(
        "Reproduction of Figure 3 (Szczepocki, 2023): Dynamic Correlation with S&P 500",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[Plot] Saved Figure 3 to: {_format_save_path(save_path)}")

    return fig


# ---------------------------------------------------------------------------
# 5. Figure 4 Reproduction: Correlations Between Candidates (Szczepocki 2023, p. 54)
# ---------------------------------------------------------------------------
def plot_figure4_candidate_correlations(
    dates: pd.Series,
    corr_filtered: np.ndarray,
    corr_q05: Optional[np.ndarray] = None,
    corr_q95: Optional[np.ndarray] = None,
    save_path: Optional[Path] = PLOTS_DIR / "fig4_reproduction_candidate_correlations.png",
) -> plt.Figure:
    """Reproduce Figure 4 (Szczepocki, 2023, p. 54): Correlations between candidate safe-haven assets.

    Pairs plotted:
    - Pair (2,3): Gold and Bitcoin
    - Pair (2,4): Gold and Ethereum
    - Pair (3,4): Bitcoin and Ethereum
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    pair_indices = [3, 4, 5]  # Indices for pairs (2,3), (2,4), (3,4) in Table 2 ordering
    pair_titles = [
        "Dynamic Correlation: Gold and Bitcoin (Pair 2-3)",
        "Dynamic Correlation: Gold and Ethereum (Pair 2-4)",
        "Dynamic Correlation: Bitcoin and Ethereum (Pair 3-4)",
    ]
    colors = ["#2CA02C", "#17BECF", "#E377C2"]

    for idx, (pair_idx, title, color) in enumerate(zip(pair_indices, pair_titles, colors)):
        ax = axes[idx]
        ax.plot(
            dates,
            corr_filtered[:, pair_idx],
            color=color,
            lw=1.2,
            label=r"Estimated $\rho_{ij,t}$ (Eq. 7)",
        )

        if corr_q05 is not None and corr_q95 is not None:
            ax.fill_between(
                dates,
                corr_q05[:, pair_idx],
                corr_q95[:, pair_idx],
                color=color,
                alpha=0.20,
                label="90% Particle Credibility Band",
            )

        ax.axhline(0.0, color="black", linestyle="--", lw=0.8, alpha=0.7)
        ax.set_ylabel(r"Correlation $\rho_t$", fontsize=9.5, fontweight="bold")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8.5)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date", fontsize=10, fontweight="bold")

    fig.suptitle(
        "Reproduction of Figure 4 (Szczepocki, 2023): Cross-Correlations Between Safe Haven Candidates",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[Plot] Saved Figure 4 to: {_format_save_path(save_path)}")

    return fig


# ---------------------------------------------------------------------------
# 6. Simulation Tracking Benchmark (Milestone 2 Validation)
# ---------------------------------------------------------------------------
def plot_simulation_tracking_benchmark(
    sim_data: Dict[str, Any],
    filter_res: Dict[str, Any],
    p: int = 4,
    save_path: Optional[Path] = PLOTS_DIR / "simulated_tracking_benchmark.png",
) -> plt.Figure:
    """Checks that the particle filter accurately recovers the true volatilities and correlations on simulated data."""
    t = np.arange(sim_data["T"])
    stdev_true = sim_data["stdev_true"]
    corr_true = sim_data["corr_true"]
    stdev_filt = filter_res["stdev_filtered"]
    corr_filt = filter_res["corr_filtered"]
    ess_rmin = filter_res.get("ESSrmin", 0.5)

    rmse_sd = np.sqrt(np.mean((stdev_filt - stdev_true) ** 2, axis=0))
    rmse_corr = np.sqrt(np.mean((corr_filt - corr_true) ** 2, axis=0))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Top-Left: Asset 1 Volatility (SPX)
    ax00 = axes[0, 0]
    ax00.plot(t, stdev_true[:, 0], "k--", lw=1.2, label="True SPX Stdev")
    ax00.plot(
        t,
        stdev_filt[:, 0],
        color="#1F4E79",
        lw=1.1,
        label=f"PF Filtered (RMSE: {rmse_sd[0]:.3f})",
    )
    if filter_res.get("stdev_q05") is not None:
        ax00.fill_between(
            t,
            filter_res["stdev_q05"][:, 0],
            filter_res["stdev_q95"][:, 0],
            color="#1F4E79",
            alpha=0.20,
            label="90% Particle Band",
        )
    ax00.set_title(
        "Simulated Volatility Tracking: Asset 1 (SPX)",
        fontweight="bold",
        fontsize=10,
    )
    ax00.set_ylabel("Stdev")
    ax00.legend(loc="upper right", fontsize=8)
    ax00.grid(True, alpha=0.25)

    # Top-Right: Asset 2 Volatility (Gold)
    ax01 = axes[0, 1]
    ax01.plot(t, stdev_true[:, 1], "k--", lw=1.2, label="True Gold Stdev")
    ax01.plot(
        t,
        stdev_filt[:, 1],
        color="#B8860B",
        lw=1.1,
        label=f"PF Filtered (RMSE: {rmse_sd[1]:.3f})",
    )
    if filter_res.get("stdev_q05") is not None:
        ax01.fill_between(
            t,
            filter_res["stdev_q05"][:, 1],
            filter_res["stdev_q95"][:, 1],
            color="#B8860B",
            alpha=0.20,
            label="90% Particle Band",
        )
    ax01.set_title(
        "Simulated Volatility Tracking: Asset 2 (Gold)",
        fontweight="bold",
        fontsize=10,
    )
    ax01.set_ylabel("Stdev")
    ax01.legend(loc="upper right", fontsize=8)
    ax01.grid(True, alpha=0.25)

    # Bottom-Left: Dynamic Correlation Pair (1,2) (SPX - Gold)
    ax10 = axes[1, 0]
    ax10.plot(t, corr_true[:, 0], "k--", lw=1.3, label="True Dynamic Corr (1,2)")
    ax10.plot(
        t,
        corr_filt[:, 0],
        color="#8E44AD",
        lw=1.1,
        label=f"PF Filtered (RMSE: {rmse_corr[0]:.3f})",
    )
    if filter_res.get("corr_q05") is not None:
        ax10.fill_between(
            t,
            filter_res["corr_q05"][:, 0],
            filter_res["corr_q95"][:, 0],
            color="#8E44AD",
            alpha=0.20,
            label="90% Particle Band",
        )
    ax10.set_title(
        "Dynamic Correlation Tracking: Pair 1-2 (SPX & Gold)",
        fontweight="bold",
        fontsize=10,
    )
    ax10.set_xlabel("Time step t")
    ax10.set_ylabel("Correlation")
    ax10.legend(loc="upper right", fontsize=8)
    ax10.grid(True, alpha=0.25)

    # Bottom-Right: ESS Profile + Tracking Summary
    ax11 = axes[1, 1]
    ax11.plot(t, filter_res["ess"], color="#2B5C8F", lw=0.9, label="Daily ESS")
    ax11.axhline(
        filter_res["N"] * ess_rmin,
        color="red",
        linestyle="--",
        lw=1.1,
        label=f"Resampling Threshold ($\\gamma N = {int(filter_res['N']*ess_rmin)}$)",
    )
    ax11.set_title(
        f"ESS Profile (Mean: {np.mean(filter_res['ess']):.1f}/{filter_res['N']}, "
        f"LogL: {filter_res['log_likelihood']:.1f})",
        fontweight="bold",
        fontsize=10,
    )
    ax11.set_xlabel("Time step t")
    ax11.set_ylabel("ESS")
    ax11.legend(loc="lower right", fontsize=8)
    ax11.grid(True, alpha=0.25)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(
            f"[Plot] Saved Simulation Tracking Benchmark to: {_format_save_path(save_path)}"
        )

    return fig


# ---------------------------------------------------------------------------
# Execution Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute full Role 2 verification: synthetic test, real filtering, benchmarks, and plot generation."""
    print("=" * 78)
    print("SMC ChMSV Project — Role 2: Bootstrap PF & Dynamic Correlations")
    print("=" * 78)

    # ---------------------------------------------------------
    # Step 1: Benchmark on Simulated 4-Asset Data (Ground Truth check)
    # ---------------------------------------------------------
    print("\n>>> Step 1: Running Particle Filter on Simulated 4-Asset Data...")
    sim4 = load_simulated_chmsv_dataset(p=4)
    model_sim = ChMSV4Asset()

    sim_res = run_bootstrap_filter(
        y_obs=sim4["y_obs"],
        model=model_sim,
        p=4,
        N=1000,
        ESSrmin=0.5,
        resampling="systematic",
        seed=42,
        compute_quantiles=True,
    )
    print(f"   [Done] Simulated Log-Likelihood: {sim_res['log_likelihood']:.4f}")
    print(f"   [Done] Mean ESS: {np.mean(sim_res['ess']):.1f} / 1000")

    fig_sim = plot_simulation_tracking_benchmark(sim4, sim_res, p=4)
    plt.close(fig_sim)

    # ---------------------------------------------------------
    # Step 2: 10-Run Variability Benchmark on Real 4-Asset Data
    # ---------------------------------------------------------
    print("\n>>> Step 2: Running 10-Run Variability Benchmark on Real 4-Asset Data...")
    real4 = load_real_dataset(p=4)
    model_real = ChMSV4Asset()

    bench_res = run_10run_variability_benchmark(
        y_obs=real4["y_obs"],
        model=model_real,
        p=4,
        N=1000,
        ESSrmin=0.5,
        n_runs=10,
        base_seed=100,
    )

    # ---------------------------------------------------------
    # Step 3: High-Precision Single Run for Figures 2, 3, and 4
    # ---------------------------------------------------------
    print("\n>>> Step 3: Generating Filtered Estimates for Figures 2, 3, and 4...")
    filt_real = run_bootstrap_filter(
        y_obs=real4["y_obs"],
        model=model_real,
        p=4,
        N=1000,
        ESSrmin=0.5,
        resampling="systematic",
        seed=42,
        compute_quantiles=True,
    )

    dates = real4["dates"]

    # Reproduce Figure 2: Return Volatilities
    fig2 = plot_figure2_volatilities(
        dates=dates,
        stdev_filtered=filt_real["stdev_filtered"],
        stdev_q05=filt_real["stdev_q05"],
        stdev_q95=filt_real["stdev_q95"],
    )
    plt.close(fig2)

    # Reproduce Figure 3: Dynamic Correlations with SPX
    fig3 = plot_figure3_spx_correlations(
        dates=dates,
        corr_filtered=filt_real["corr_filtered"],
        corr_q05=filt_real["corr_q05"],
        corr_q95=filt_real["corr_q95"],
    )
    plt.close(fig3)

    # Reproduce Figure 4: Candidate Safe Haven Cross-Correlations
    fig4 = plot_figure4_candidate_correlations(
        dates=dates,
        corr_filtered=filt_real["corr_filtered"],
        corr_q05=filt_real["corr_q05"],
        corr_q95=filt_real["corr_q95"],
    )
    plt.close(fig4)

    print("\n" + "=" * 78)
    print("Role 2 Pipeline Complete! All benchmark metrics computed and plots saved to `plots/`.")
    print("=" * 78)


if __name__ == "__main__":
    main()
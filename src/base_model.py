"""
Phase 1: Data & Base Setup (Data & Architecture Lead - Gabriel)

This module implements:
1. The base `StochVol` (`StochasticVolatility`) StateSpaceModel class in `particles`
   following Chopin & Papaspiliopoulos (2020), Section 2.4.3 & Chapter 4 (p. 47).
2. Synthetic sandbox data simulation (fake log-returns + ground-truth hidden volatility)
   and persistence to `data/processed/simulated_sv_data.csv`.
3. Real-world S&P 500 data ingestion, cleaning, and log-return computation
   saved to `data/processed/sp500_log_returns.csv` and `data/processed/sp500_full_cleaned.csv`.
4. Baseline Bootstrap particle filter verification, evaluation metrics (ESS, MSE, RMSE),
   and parallel execution via `particles.multiSMC`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import particles
from particles import collectors
from particles import distributions as dists
from particles import state_space_models as ssm

# ---------------------------------------------------------------------------
# Project Paths & Default Ground-Truth Sandbox Parameters
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "plots"

SIMULATED_CSV_PATH = DATA_PROCESSED_DIR / "simulated_sv_data.csv"
SIMULATED_META_PATH = DATA_PROCESSED_DIR / "simulated_sv_metadata.json"
SP500_RAW_CSV_PATH = DATA_RAW_DIR / "sp500_raw.csv"
SP500_CLEANED_CSV_PATH = DATA_PROCESSED_DIR / "sp500_log_returns.csv"
SP500_FULL_CLEANED_CSV_PATH = DATA_PROCESSED_DIR / "sp500_full_cleaned.csv"

# Ground-truth parameters for the synthetic sandbox dataset:
# Matches realistic financial volatility clustering (Chopin & Papaspiliopoulos, 2020)
# Provides both 'rho' (book notation) and 'phi' (milestone table notation),
# as well as 'sigma' and 'sigma2'.
TRUE_SIM_PARAMS: Dict[str, Any] = {
    "mu": -1.0,
    "rho": 0.95,
    "phi": 0.95,
    "sigma": 0.2,
    "sigma2": 0.04,
    "T": 500,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# 1. Base State-Space Model Class in `particles`
# ---------------------------------------------------------------------------
class StochVol(ssm.StateSpaceModel):
    r"""Univariate Stochastic Volatility state-space model.

    Mathematical formulation (Chopin & Papaspiliopoulos, 2020, Sec. 2.4.3 & Ch. 4):
    .. math::
        X_0 &\sim \mathcal{N}\left(\mu, \frac{\sigma^2}{1 - \phi^2}\right) \\
        X_t \mid X_{t-1} = x_{t-1} &\sim \mathcal{N}\left(\mu + \phi (x_{t-1} - \mu), \sigma^2\right), \quad t \ge 1 \\
        Y_t \mid X_t = x_t &\sim \mathcal{N}\left(0, \exp(x_t)\right), \quad t \ge 0

    where:
    - :math:`Y_t` is the observed asset log-return at time :math:`t`,
    - :math:`X_t` is the unobserved (latent) log-volatility state,
    - :math:`\theta = (\mu, \phi, \sigma)` (or :math:`(\mu, \rho, \sigma)`) are the
      model parameters with stationarity condition :math:`|\phi| < 1` and :math:`\sigma > 0`.

    Notes for Teammates:
    --------------------
    - Parameter aliasing: Both `rho` (used in Chopin's book) and `phi` (used in our
      project milestone sheet) are accepted and kept synchronized, as are `sigma`
      and `sigma2`.
    """

    default_params = {
        "mu": TRUE_SIM_PARAMS["mu"],
        "rho": TRUE_SIM_PARAMS["rho"],
        "phi": TRUE_SIM_PARAMS["phi"],
        "sigma": TRUE_SIM_PARAMS["sigma"],
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Synchronize 'phi' <-> 'rho' so either parameter name works seamlessly
        if "phi" in kwargs and "rho" not in kwargs:
            self.rho = float(kwargs["phi"])
            self.phi = float(kwargs["phi"])
        elif "rho" in kwargs and "phi" not in kwargs:
            self.phi = float(kwargs["rho"])
            self.rho = float(kwargs["rho"])
        else:
            self.rho = float(getattr(self, "rho", self.default_params["rho"]))
            self.phi = float(getattr(self, "phi", self.rho))

        # Synchronize 'sigma2' <-> 'sigma' so priors over sigma^2 or sigma both work
        if "sigma2" in kwargs and "sigma" not in kwargs:
            self.sigma2 = float(kwargs["sigma2"])
            self.sigma = float(np.sqrt(max(self.sigma2, 1e-12)))
        else:
            self.sigma = float(getattr(self, "sigma", self.default_params["sigma"]))
            self.sigma2 = float(self.sigma**2)

        self.mu = float(getattr(self, "mu", self.default_params["mu"]))

    def sig0(self) -> float:
        """Stationary standard deviation of initial state X_0."""
        denom = max(1.0 - self.rho**2, 1e-12)
        return float(self.sigma / np.sqrt(denom))

    def EXt(self, xp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Conditional expectation E[X_t | X_{t-1} = xp]."""
        return self.mu + self.rho * (xp - self.mu)

    def PX0(self) -> dists.ProbDist:
        """Initial state distribution P_0(dx_0) = N(mu, sigma^2 / (1 - rho^2))."""
        return dists.Normal(loc=self.mu, scale=self.sig0())

    def PX(self, t: int, xp: Union[float, np.ndarray]) -> dists.ProbDist:
        """State transition distribution P_t(x_{t-1}, dx_t) = N(mu + rho*(x_{t-1} - mu), sigma^2)."""
        return dists.Normal(loc=self.EXt(xp), scale=self.sigma)

    def PY(
        self,
        t: int,
        xp: Optional[Union[float, np.ndarray]],
        x: Union[float, np.ndarray],
    ) -> dists.ProbDist:
        """Observation emission distribution F_t(x_t, dy_t) = N(0, exp(x_t))."""
        return dists.Normal(loc=0.0, scale=np.exp(0.5 * x))


# Alias for convenience so teammates can import either StochVol or StochasticVolatility
StochasticVolatility = StochVol


# ---------------------------------------------------------------------------
# 2. Synthetic Sandbox Data Simulation & Loading (M1 & M2)
# ---------------------------------------------------------------------------
def generate_simulated_dataset(
    T: int = TRUE_SIM_PARAMS["T"],
    mu: float = TRUE_SIM_PARAMS["mu"],
    rho: float = TRUE_SIM_PARAMS["rho"],
    sigma: float = TRUE_SIM_PARAMS["sigma"],
    seed: int = TRUE_SIM_PARAMS["seed"],
    save_csv: bool = True,
    csv_path: Path = SIMULATED_CSV_PATH,
    meta_path: Path = SIMULATED_META_PATH,
) -> pd.DataFrame:
    """Simulate synthetic log-returns and true hidden volatility from `StochVol`.

    Parameters
    ----------
    T : int
        Number of time steps to simulate (default: 500).
    mu : float
        Long-run mean of log-volatility (default: -1.0).
    rho : float
        AR(1) persistence parameter, |rho| < 1 (default: 0.95).
    sigma : float
        Standard deviation of state innovation noise (default: 0.2).
    seed : int
        Random seed for reproducibility (default: 42).
    save_csv : bool
        Whether to save the generated dataset to `data/processed/simulated_sv_data.csv`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - `t`: integer time index (0 to T-1)
        - `x_true`: true hidden log-volatility state X_t
        - `sigma_true`: true hidden volatility scale exp(0.5 * X_t)
        - `var_true`: true hidden variance exp(X_t)
        - `y_obs`: observed synthetic log-return Y_t
    """
    np.random.seed(seed)
    model = StochVol(mu=mu, rho=rho, sigma=sigma)
    x_list, y_list = model.simulate(T)

    x_true = np.asarray(x_list, dtype=float).ravel()
    y_obs = np.asarray(y_list, dtype=float).ravel()
    sigma_true = np.exp(0.5 * x_true)
    var_true = np.exp(x_true)

    df = pd.DataFrame(
        {
            "t": np.arange(T, dtype=int),
            "x_true": x_true,
            "sigma_true": sigma_true,
            "var_true": var_true,
            "y_obs": y_obs,
        }
    )

    if save_csv:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        metadata = {
            "mu": mu,
            "rho": rho,
            "phi": rho,
            "sigma": sigma,
            "sigma2": round(sigma**2, 6),
            "T": T,
            "seed": seed,
            "description": (
                "Synthetic Stochastic Volatility sandbox dataset: "
                "X_t = mu + phi*(X_{t-1} - mu) + sigma*U_t, Y_t = exp(X_t/2)*V_t"
            ),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return df


def load_simulated_dataset(
    csv_path: Path = SIMULATED_CSV_PATH,
) -> Dict[str, Any]:
    """Load the simulated sandbox dataset (generating it if not yet on disk).

    Returns
    -------
    dict
        Dictionary containing:
        - `'df'`: full pandas DataFrame
        - `'x_true'`: 1D numpy array of true hidden log-volatility X_t
        - `'sigma_true'`: 1D numpy array of true volatility exp(X_t / 2)
        - `'y_obs'`: 1D numpy array of observed returns Y_t (ready for `particles`)
        - `'params'`: dictionary of ground-truth parameters (mu, rho, phi, sigma, sigma2)
    """
    if not csv_path.exists():
        df = generate_simulated_dataset(save_csv=True, csv_path=csv_path)
    else:
        df = pd.read_csv(csv_path)

    params = dict(TRUE_SIM_PARAMS)
    if SIMULATED_META_PATH.exists():
        with open(SIMULATED_META_PATH, "r", encoding="utf-8") as f:
            params.update(json.load(f))

    return {
        "df": df,
        "x_true": df["x_true"].to_numpy(dtype=float),
        "sigma_true": df["sigma_true"].to_numpy(dtype=float),
        "y_obs": df["y_obs"].to_numpy(dtype=float),
        "params": params,
    }


# ---------------------------------------------------------------------------
# 3. Real-World S&P 500 Data Pipeline (M1 & M3)
# ---------------------------------------------------------------------------
def _read_raw_sp500_csv(raw_csv_path: Path) -> pd.DataFrame:
    """Read raw S&P 500 CSV, handling both MultiIndex yfinance headers and flat CSVs."""
    with open(raw_csv_path, "r", encoding="utf-8") as f:
        first_lines = [f.readline().strip() for _ in range(3)]

    if first_lines[0].startswith("Price,") and first_lines[1].startswith("Ticker,"):
        df = pd.read_csv(raw_csv_path, skiprows=[1, 2])
        if "Price" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"Price": "Date"})
    else:
        df = pd.read_csv(raw_csv_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).reset_index(drop=True)
    return df


def prepare_sp500_dataset(
    raw_csv_path: Path = SP500_RAW_CSV_PATH,
    processed_csv_path: Path = SP500_CLEANED_CSV_PATH,
    full_processed_csv_path: Path = SP500_FULL_CLEANED_CSV_PATH,
    start_date: Optional[str] = "2019-07-01",
    end_date: Optional[str] = "2021-06-30",
    download_if_missing: bool = False,
) -> pd.DataFrame:
    """Clean raw S&P 500 price series and compute daily log-returns.

    Following Chopin & Papaspiliopoulos (2020, Sec. 2.4.3 & Sec. 10.5.2), we compute:
    - `log_return`: raw log-return log(P_t / P_{t-1})
    - `pct_log_return`: percentage daily log-return 100 * log(P_t / P_{t-1})
    - `demeaned_pct_log_return`: mean-zero percentage daily log-return (standard input
      for the zero-mean observation equation Y_t | X_t ~ N(0, exp(X_t)) with mu ~ -1)
    - `rolling_vol_20d`: 20-day rolling standard deviation of percentage log-returns

    The default window (2019-07-01 to 2021-06-30, ~504 trading days) captures the
    2020 COVID-19 market crash and recovery, providing an ideal benchmark for
    live filtering, hindsight smoothing (M3 crash reconstruction), and parameter inference.
    """
    if not raw_csv_path.exists():
        if download_if_missing:
            import yfinance as yf

            raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
            raw_df = yf.download("^GSPC", start="2018-01-01", end="2025-01-01", progress=False)
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.get_level_values(0)
            raw_df = raw_df.reset_index()
            raw_df.to_csv(raw_csv_path, index=False)
        else:
            raise FileNotFoundError(
                f"Raw S&P 500 file not found at {raw_csv_path}. "
                "Set download_if_missing=True to fetch via yfinance."
            )

    df = _read_raw_sp500_csv(raw_csv_path)
    # Normalize raw CSV on disk to clean single-header format for teammates
    df.to_csv(raw_csv_path, index=False)

    # Compute log-returns: Y_t = log(P_t / P_{t-1})
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["pct_log_return"] = 100.0 * df["log_return"]
    df["rolling_vol_20d"] = df["pct_log_return"].rolling(window=20).std()

    # Drop the initial NaN row from differencing
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    df["demeaned_pct_log_return"] = df["pct_log_return"] - df["pct_log_return"].mean()
    df["t"] = np.arange(len(df), dtype=int)

    # Save full multi-year cleaned dataset
    full_processed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_processed_csv_path, index=False)

    # Slice the target window (e.g., 2019-07-01 to 2021-06-30 covering the 2020 crash)
    sub_df = df.copy()
    if start_date is not None:
        sub_df = sub_df[sub_df["Date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        sub_df = sub_df[sub_df["Date"] <= pd.to_datetime(end_date)]
    sub_df = sub_df.reset_index(drop=True)
    sub_df["demeaned_pct_log_return"] = (
        sub_df["pct_log_return"] - sub_df["pct_log_return"].mean()
    )
    sub_df["t"] = np.arange(len(sub_df), dtype=int)

    processed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(processed_csv_path, index=False)
    return sub_df


def load_sp500_dataset(
    csv_path: Path = SP500_CLEANED_CSV_PATH,
    return_col: str = "demeaned_pct_log_return",
) -> Dict[str, Any]:
    """Load the cleaned S&P 500 log-returns dataset.

    Parameters
    ----------
    csv_path : Path
        Path to the processed S&P 500 CSV (`data/processed/sp500_log_returns.csv`).
    return_col : str
        Column to extract as `y_obs` for `particles` models. Defaults to
        `'demeaned_pct_log_return'` (100 * demeaned log-returns, matching the scale
        of Chopin & Papaspiliopoulos 2020, Sec. 10.5.2).

    Returns
    -------
    dict
        Dictionary containing:
        - `'df'`: pandas DataFrame with Date, Close, log_return, pct_log_return, etc.
        - `'dates'`: pandas Series of timestamps
        - `'prices'`: 1D numpy array of S&P 500 closing prices
        - `'y_obs'`: 1D numpy array of log-returns ready for `particles.SMC`
    """
    if not csv_path.exists():
        df = prepare_sp500_dataset(processed_csv_path=csv_path)
    else:
        df = pd.read_csv(csv_path, parse_dates=["Date"])

    return {
        "df": df,
        "dates": pd.to_datetime(df["Date"]),
        "prices": df["Close"].to_numpy(dtype=float),
        "y_obs": df[return_col].to_numpy(dtype=float),
    }


# ---------------------------------------------------------------------------
# 4. Baseline Bootstrap Filter, Evaluation Metrics & Parallel Runner (M2 & M3)
# ---------------------------------------------------------------------------
def evaluate_filter_performance(
    x_est: np.ndarray,
    ess_trace: np.ndarray,
    x_true: Optional[np.ndarray] = None,
    N: Optional[int] = None,
    log_likelihood: Optional[float] = None,
) -> Dict[str, float]:
    """Compute standardized filter evaluation metrics (ESS, MSE, RMSE, Log-Likelihood).

    Used across Role 1 (Baseline Bootstrap) and Role 2 (Guided PF & APF benchmarks).
    """
    x_est = np.asarray(x_est, dtype=float).ravel()
    ess_trace = np.asarray(ess_trace, dtype=float).ravel()

    metrics: Dict[str, float] = {
        "mean_ess": float(np.mean(ess_trace)),
        "min_ess": float(np.min(ess_trace)),
        "median_ess": float(np.median(ess_trace)),
    }
    if N is not None and N > 0:
        metrics["ess_pct"] = float(100.0 * np.mean(ess_trace) / N)
    if log_likelihood is not None:
        metrics["log_likelihood"] = float(log_likelihood)

    if x_true is not None:
        x_true = np.asarray(x_true, dtype=float).ravel()
        vol_true = np.exp(0.5 * x_true)
        vol_est = np.exp(0.5 * x_est)
        metrics["mse_x"] = float(np.mean((x_est - x_true) ** 2))
        metrics["rmse_x"] = float(np.sqrt(metrics["mse_x"]))
        metrics["mae_x"] = float(np.mean(np.abs(x_est - x_true)))
        metrics["mse_vol"] = float(np.mean((vol_est - vol_true) ** 2))
        metrics["rmse_vol"] = float(np.sqrt(metrics["mse_vol"]))

    return metrics


def run_baseline_bootstrap(
    y_data: np.ndarray,
    model: Optional[StochVol] = None,
    N: int = 1000,
    resampling: str = "systematic",
    ESSrmin: float = 0.5,
    store_history: bool = False,
    qmc: bool = False,
    seed: Optional[int] = 42,
    x_true: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run the baseline Bootstrap particle filter on `y_data`.

    Parameters
    ----------
    y_data : np.ndarray
        1D array of observed log-returns Y_{0:T-1}.
    model : StochVol, optional
        Instance of `StochVol`. Defaults to `StochVol()` with `TRUE_SIM_PARAMS`.
    N : int
        Number of particles (default: 1000).
    resampling : str
        Resampling scheme ('systematic', 'stratified', 'residual', 'multinomial').
    ESSrmin : float
        Adaptive resampling threshold as a fraction of N (default: 0.5).
    store_history : bool
        Set True to store full particle history `pf.hist` for Role 3 (Smoothing).
    qmc : bool
        Set True to run Sequential Quasi-Monte Carlo (SQMC).
    seed : int, optional
        Random seed for reproducibility.
    x_true : np.ndarray, optional
        Ground-truth log-volatility X_{0:T-1} (if available) to compute MSE/RMSE.

    Returns
    -------
    dict
        Dictionary with filtered trajectories, ESS, log-likelihoods, and metrics.
    """
    if seed is not None:
        np.random.seed(seed)

    if model is None:
        model = StochVol()

    y_data = np.asarray(y_data, dtype=float).ravel()
    fk_boot = ssm.Bootstrap(ssm=model, data=y_data)

    pf = particles.SMC(
        fk=fk_boot,
        N=N,
        resampling=resampling,
        ESSrmin=ESSrmin,
        store_history=store_history,
        qmc=qmc,
        collect=[collectors.Moments()],
    )
    pf.run()

    x_filt_mean = np.asarray([m["mean"] for m in pf.summaries.moments], dtype=float).ravel()
    x_filt_var = np.asarray([m["var"] for m in pf.summaries.moments], dtype=float).ravel()
    x_filt_std = np.sqrt(np.maximum(x_filt_var, 0.0))
    # Log-normal moment approximation for E[exp(0.5 * X_t) | Y_{0:t}]
    vol_filt_mean = np.exp(0.5 * x_filt_mean + 0.125 * x_filt_var)

    ess_trace = np.asarray(pf.summaries.ESSs, dtype=float).ravel()
    logLts = np.asarray(pf.summaries.logLts, dtype=float).ravel()
    final_log_lik = float(logLts[-1])

    metrics = evaluate_filter_performance(
        x_est=x_filt_mean,
        ess_trace=ess_trace,
        x_true=x_true,
        N=N,
        log_likelihood=final_log_lik,
    )

    return {
        "pf": pf,
        "fk": fk_boot,
        "model": model,
        "x_filt_mean": x_filt_mean,
        "x_filt_var": x_filt_var,
        "x_filt_std": x_filt_std,
        "vol_filt_mean": vol_filt_mean,
        "ess": ess_trace,
        "logLts": logLts,
        "final_log_likelihood": final_log_lik,
        "metrics": metrics,
    }


def _default_multismc_summary(pf: particles.SMC) -> Dict[str, Any]:
    """Memory-efficient summary function for parallel `multiSMC` runs."""
    x_mean = np.asarray([m["mean"] for m in pf.summaries.moments], dtype=float).ravel()
    x_var = np.asarray([m["var"] for m in pf.summaries.moments], dtype=float).ravel()
    ess = np.asarray(pf.summaries.ESSs, dtype=float).ravel()
    logLts = np.asarray(pf.summaries.logLts, dtype=float).ravel()
    return {
        "x_filt_mean": x_mean,
        "x_filt_var": x_var,
        "ess": ess,
        "logL": float(logLts[-1]),
    }


def run_parallel_smc(
    y_data: np.ndarray,
    model: Optional[StochVol] = None,
    fk_cls: Union[Any, Dict[str, Any]] = ssm.Bootstrap,
    N: Union[int, List[int]] = 1000,
    nruns: int = 8,
    nprocs: int = 0,
    resampling: Union[str, List[str]] = "systematic",
    ESSrmin: float = 0.5,
    x_true: Optional[np.ndarray] = None,
    out_func: Optional[Callable[[particles.SMC], Any]] = None,
) -> Dict[str, Any]:
    """Run multiple SMC particle filters in parallel across CPU cores via `particles.multiSMC`.

    Parameters
    ----------
    y_data : np.ndarray
        Observed log-returns array.
    model : StochVol, optional
        StateSpaceModel instance (defaults to `StochVol()`).
    fk_cls : FeynmanKac class or dict of {name: FeynmanKac}
        Either a single FeynmanKac class (e.g., `ssm.Bootstrap`) or a dictionary of
        FeynmanKac objects/classes to benchmark in parallel (e.g., Bootstrap vs Guided vs APF).
    N : int or list of int
        Number of particles per filter run.
    nruns : int
        Number of independent Monte Carlo runs per configuration (default: 8).
    nprocs : int
        Number of CPU cores to use (`0` uses all available cores via `multiprocessing`).
    resampling : str or list of str
        Resampling algorithm(s).
    ESSrmin : float
        ESS threshold for adaptive resampling.
    x_true : np.ndarray, optional
        True hidden log-volatility (if benchmarking on synthetic data).
    out_func : callable, optional
        Custom output extractor for each `SMC` run.

    Returns
    -------
    dict
        Aggregated parallel run statistics (mean trajectory, IQR, ESS, MSE, logL std).
    """
    if model is None:
        model = StochVol()

    y_data = np.asarray(y_data, dtype=float).ravel()
    summary_fn = out_func if out_func is not None else _default_multismc_summary

    if isinstance(fk_cls, dict):
        fk_arg = fk_cls
    else:
        fk_arg = fk_cls(ssm=model, data=y_data)

    try:
        raw_results = particles.multiSMC(
            fk=fk_arg,
            N=N,
            nruns=nruns,
            nprocs=nprocs,
            resampling=resampling,
            ESSrmin=ESSrmin,
            collect=[collectors.Moments()],
            out_func=summary_fn,
        )
    except Exception:
        # Fallback to single-process execution if OS multiprocessing fork is restricted
        raw_results = particles.multiSMC(
            fk=fk_arg,
            N=N,
            nruns=nruns,
            nprocs=1,
            resampling=resampling,
            ESSrmin=ESSrmin,
            collect=[collectors.Moments()],
            out_func=summary_fn,
        )

    if out_func is not None:
        return {"raw_results": raw_results}

    def _get_field(r: Dict[str, Any], key: str) -> Any:
        return r[key] if key in r else r["output"][key]

    x_means = np.stack([_get_field(r, "x_filt_mean") for r in raw_results], axis=0)
    x_vars = np.stack([_get_field(r, "x_filt_var") for r in raw_results], axis=0)
    ess_runs = np.stack([_get_field(r, "ess") for r in raw_results], axis=0)
    logLs = np.asarray([_get_field(r, "logL") for r in raw_results], dtype=float)

    mean_x_trajectory = np.mean(x_means, axis=0)
    q25_x_trajectory = np.percentile(x_means, 25, axis=0)
    q75_x_trajectory = np.percentile(x_means, 75, axis=0)
    mean_vol_trajectory = np.mean(np.exp(0.5 * x_means + 0.125 * x_vars), axis=0)
    mean_ess_trajectory = np.mean(ess_runs, axis=0)

    summary: Dict[str, Any] = {
        "nruns": len(raw_results),
        "x_means_all": x_means,
        "x_filt_mean": mean_x_trajectory,
        "x_filt_q25": q25_x_trajectory,
        "x_filt_q75": q75_x_trajectory,
        "vol_filt_mean": mean_vol_trajectory,
        "ess_mean_trace": mean_ess_trajectory,
        "mean_ess": float(np.mean(ess_runs)),
        "std_ess": float(np.std( np.mean(ess_runs, axis=1) )),
        "logL_mean": float(np.mean(logLs)),
        "logL_std": float(np.std(logLs)),
        "logL_all": logLs,
        "raw_results": raw_results,
    }

    if x_true is not None:
        x_true_arr = np.asarray(x_true, dtype=float).ravel()
        mses = np.mean((x_means - x_true_arr[None, :]) ** 2, axis=1)
        rmses = np.sqrt(mses)
        summary["mse_mean"] = float(np.mean(mses))
        summary["mse_std"] = float(np.std(mses))
        summary["rmse_mean"] = float(np.mean(rmses))
        summary["rmse_std"] = float(np.std(rmses))

    return summary


# ---------------------------------------------------------------------------
# 5. Diagnostic & Milestone Visualization Plots
# ---------------------------------------------------------------------------
def plot_sandbox_validation(
    sim_data: Dict[str, Any],
    boot_res: Dict[str, Any],
    save_path: Optional[Path] = PLOTS_DIR / "m2_sandbox_bootstrap_validation.png",
) -> plt.Figure:
    """Plot Milestone 2 Sandbox Validation: Simulated data + Bootstrap PF tracking & ESS."""
    x_true = sim_data["x_true"]
    sigma_true = sim_data["sigma_true"]
    y_obs = sim_data["y_obs"]
    T = len(y_obs)
    t_grid = np.arange(T)

    x_filt = boot_res["x_filt_mean"]
    x_std = boot_res["x_filt_std"]
    ess = boot_res["ess"]
    metrics = boot_res["metrics"]
    N = boot_res["pf"].N

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Panel 1: Simulated Returns & True Volatility Envelope
    ax0 = axes[0]
    ax0.plot(t_grid, y_obs, color="#4C72B0", alpha=0.65, lw=0.9, label="Simulated Log-Returns $Y_t$")
    ax0.plot(t_grid, 2.0 * sigma_true, color="#C44E52", lw=1.3, ls="--", label=r"$\pm 2\sigma_t = \pm 2\exp(X_t/2)$ Envelope")
    ax0.plot(t_grid, -2.0 * sigma_true, color="#C44E52", lw=1.3, ls="--")
    ax0.set_ylabel("Log-Return $Y_t$")
    ax0.set_title(
        f"Milestone 2 Sandbox Validation: Synthetic SV Data ($T={T}$, "
        f"$\\mu={sim_data['params']['mu']}$, $\\phi={sim_data['params']['phi']}$, "
        f"$\\sigma={sim_data['params']['sigma']}$) & Baseline Bootstrap Filter ($N={N}$)",
        fontsize=12,
        fontweight="bold",
    )
    ax0.legend(loc="upper right", frameon=True)
    ax0.grid(True, alpha=0.25)

    # Panel 2: Hidden Log-Volatility Tracking (True vs. Bootstrap Filter)
    ax1 = axes[1]
    ax1.plot(t_grid, x_true, color="#222222", lw=1.5, label="True Hidden Log-Volatility $X_t$")
    ax1.plot(
        t_grid,
        x_filt,
        color="#DD8452",
        lw=1.4,
        label=f"Bootstrap PF $\\mathbb{{E}}[X_t \\mid Y_{{0:t}}]$ (RMSE = {metrics['rmse_x']:.3f}, MSE = {metrics['mse_x']:.4f})",
    )
    ax1.fill_between(
        t_grid,
        x_filt - 1.96 * x_std,
        x_filt + 1.96 * x_std,
        color="#DD8452",
        alpha=0.22,
        label="95% Filtering Credible Band",
    )
    ax1.set_ylabel("Log-Volatility $X_t$")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.25)

    # Panel 3: Effective Sample Size (ESS)
    ax2 = axes[2]
    ax2.plot(t_grid, ess, color="#55A868", lw=1.2, label=f"Effective Sample Size (Mean = {metrics['mean_ess']:.1f})")
    ax2.axhline(
        0.5 * N,
        color="#C44E52",
        ls="--",
        lw=1.2,
        label=f"Resampling Threshold ($\\mathrm{{ESS}}_{{\\min}} = {int(0.5 * N)}$)",
    )
    ax2.set_ylim(0, N * 1.05)
    ax2.set_ylabel("ESS")
    ax2.set_xlabel("Time Step $t$ (Trading Days)")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def plot_sp500_overview_and_baseline(
    sp500_data: Dict[str, Any],
    boot_res: Dict[str, Any],
    save_path: Optional[Path] = PLOTS_DIR / "m3_sp500_overview_and_baseline.png",
) -> plt.Figure:
    """Plot Milestone 3 Real-World S&P 500 Dataset & Baseline Volatility Tracking."""
    df = sp500_data["df"]
    dates = sp500_data["dates"]
    prices = sp500_data["prices"]
    y_obs = sp500_data["y_obs"]
    vol_filt = boot_res["vol_filt_mean"]
    metrics = boot_res["metrics"]
    N = boot_res["pf"].N

    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True)

    # Panel 1: S&P 500 Price Level
    ax0 = axes[0]
    ax0.plot(dates, prices, color="#1F4E79", lw=1.6, label="S&P 500 Index (^GSPC Close)")
    ax0.axvspan(
        pd.Timestamp("2020-02-19"),
        pd.Timestamp("2020-04-15"),
        color="#C44E52",
        alpha=0.18,
        label="COVID-19 Market Crash (Feb–Apr 2020)",
    )
    ax0.set_ylabel("Index Level")
    ax0.set_title(
        f"Milestone 3 Real-World Dataset: S&P 500 Log-Returns ($T={len(y_obs)}$) & Baseline Volatility Tracker ($N={N}$)",
        fontsize=12,
        fontweight="bold",
    )
    ax0.legend(loc="upper left", frameon=True)
    ax0.grid(True, alpha=0.25)

    # Panel 2: Cleaned Daily Percentage Log-Returns
    ax1 = axes[1]
    ax1.plot(dates, y_obs, color="#4C72B0", lw=0.9, alpha=0.8, label="Cleaned Daily Log-Returns $Y_t = 100 \\log(P_t/P_{t-1})$")
    ax1.set_ylabel("Log-Return (%)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.25)

    # Panel 3: Live Volatility Tracking vs Rolling Realized Volatility
    ax2 = axes[2]
    if "rolling_vol_20d" in df.columns:
        ax2.plot(
            dates,
            df["rolling_vol_20d"],
            color="#888888",
            lw=1.3,
            ls="--",
            label="20-Day Rolling Realized Volatility (%)",
        )
    ax2.plot(
        dates,
        vol_filt,
        color="#C44E52",
        lw=1.5,
        label=f"Bootstrap PF Filtered Volatility $\\mathbb{{E}}[\\exp(X_t/2) \\mid Y_{{0:t}}]$ (Mean ESS = {metrics['mean_ess']:.0f})",
    )
    ax2.set_ylabel("Daily Volatility (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# CLI Entry Point for Phase 1 Verification
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate simulated & S&P 500 datasets, verify baseline Bootstrap PF, and save plots."""
    print("=" * 72)
    print("SMC Stochastic Volatility — Phase 1: Data & Base Setup Pipeline")
    print("=" * 72)

    # 1. Generate and load synthetic sandbox data (M1 & M2)
    sim_df = generate_simulated_dataset(save_csv=True)
    sim_data = load_simulated_dataset()
    print(
        f"[M1/M2] Simulated sandbox dataset saved to: {SIMULATED_CSV_PATH.relative_to(PROJECT_ROOT)} "
        f"(shape: {sim_df.shape}, params: mu={TRUE_SIM_PARAMS['mu']}, "
        f"phi={TRUE_SIM_PARAMS['phi']}, sigma={TRUE_SIM_PARAMS['sigma']})"
    )

    # 2. Verify baseline Bootstrap filter & parallel multiSMC on synthetic data (M2)
    boot_sim = run_baseline_bootstrap(
        y_data=sim_data["y_obs"],
        model=StochVol(**TRUE_SIM_PARAMS),
        N=1000,
        x_true=sim_data["x_true"],
    )
    m_sim = boot_sim["metrics"]
    print(
        f"[M2] Baseline Bootstrap PF on Sandbox Data (N=1000): "
        f"RMSE(X) = {m_sim['rmse_x']:.4f}, MSE(X) = {m_sim['mse_x']:.4f}, "
        f"Mean ESS = {m_sim['mean_ess']:.1f} ({m_sim['ess_pct']:.1f}%), "
        f"logL = {m_sim['log_likelihood']:.2f}"
    )

    par_sim = run_parallel_smc(
        y_data=sim_data["y_obs"],
        model=StochVol(**TRUE_SIM_PARAMS),
        N=1000,
        nruns=8,
        nprocs=0,
        x_true=sim_data["x_true"],
    )
    print(
        f"[M2/M3] Parallel multiSMC (8 runs across CPU cores): "
        f"Mean RMSE(X) = {par_sim['rmse_mean']:.4f} +/- {par_sim['rmse_std']:.4f}, "
        f"Mean logL = {par_sim['logL_mean']:.2f} +/- {par_sim['logL_std']:.2f}"
    )

    fig_sim = plot_sandbox_validation(sim_data, boot_sim)
    plt.close(fig_sim)
    print(f"[M2] Saved sandbox validation plot to: plots/m2_sandbox_bootstrap_validation.png")

    # 3. Clean S&P 500 dataset and run baseline filter on real market data (M3)
    sp500_df = prepare_sp500_dataset()
    sp500_data = load_sp500_dataset()
    print(
        f"[M3] Cleaned S&P 500 log-returns saved to: {SP500_CLEANED_CSV_PATH.relative_to(PROJECT_ROOT)} "
        f"(T={len(sp500_df)} trading days, {sp500_df['Date'].iloc[0].strftime('%Y-%m-%d')} "
        f"to {sp500_df['Date'].iloc[-1].strftime('%Y-%m-%d')})"
    )

    boot_sp500 = run_baseline_bootstrap(
        y_data=sp500_data["y_obs"],
        model=StochVol(mu=-0.8, rho=0.97, sigma=0.25),
        N=1000,
    )
    m_sp = boot_sp500["metrics"]
    print(
        f"[M3] Baseline Bootstrap PF on S&P 500 (N=1000): "
        f"Mean ESS = {m_sp['mean_ess']:.1f} ({m_sp['ess_pct']:.1f}%), "
        f"Min ESS = {m_sp['min_ess']:.1f}, logL = {m_sp['log_likelihood']:.2f}"
    )

    fig_sp = plot_sp500_overview_and_baseline(sp500_data, boot_sp500)
    plt.close(fig_sp)
    print(f"[M3] Saved S&P 500 overview & baseline plot to: plots/m3_sp500_overview_and_baseline.png")
    print("=" * 72)


if __name__ == "__main__":
    main()

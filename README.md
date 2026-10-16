# SMC Stochastic Volatility Project

This repository contains the codebase for our December 17 project on **Stochastic Volatility** using the [`particles`](https://github.com/nchopin/particles) package in Python (accompanying *An Introduction to Sequential Monte Carlo* by Nicolas Chopin & Omiros Papaspiliopoulos, 2020).

---

## Project Structure

```text
├── data/
│   ├── raw/
│   │   └── sp500_raw.csv                  <- Original S&P 500 (^GSPC) daily OHLCV market data
│   └── processed/
│       ├── simulated_sv_data.csv          <- Synthetic sandbox dataset (y_obs + ground-truth x_true)
│       ├── simulated_sv_metadata.json     <- Ground-truth parameters (mu=-1.0, phi=0.95, sigma=0.2)
│       ├── sp500_log_returns.csv          <- Cleaned S&P 500 percentage log-returns (2019–2021 COVID crash window)
│       └── sp500_full_cleaned.csv         <- Full multi-year cleaned S&P 500 dataset (2018–2024)
├── notebooks/
│   └── smc_stochastic_volatility.ipynb    <- Shared main Jupyter notebook (Math + Code + Plots for all 4 roles)
├── plots/
│   ├── m2_sandbox_bootstrap_validation.png <- M2 Sandbox proof: Simulated data + Bootstrap PF tracking & ESS
│   └── m3_sp500_overview_and_baseline.png  <- M3 Real-market proof: S&P 500 crash data + Baseline volatility tracking
├── src/
│   ├── __init__.py                        <- Package exports for models, data loaders, and parallel runners
│   ├── base_model.py                      <- Role 1: Data & Base Setup (Gabriel)
│   ├── filtering.py                       <- Role 2: Live Tracking / Filtering — Bootstrap, Guided, APF (Volunteer 1)
│   ├── smoothing.py                       <- Role 3: Hindsight Smoothing — FFBS & Two-Filter (Volunteer 2)
│   └── estimation.py                      <- Role 4: Parameter Estimation — PMCMC & SMC^2 (Volunteer 3)
├── run_pipeline.py                        <- Parallelized end-to-end runner script using `particles.multiSMC`
├── requirements.txt                       <- Python dependencies
└── README.md                              <- Developer guide & milestone overview
```

---

## Mathematical Model (`StochVol`)

Implemented in `src/base_model.py` (`StochVol` / `StochasticVolatility`), following Chopin & Papaspiliopoulos (2020, Section 2.4.3 & Chapter 4):

$$\begin{aligned}
X_0 &\sim \mathcal{N}\!\left(\mu, \frac{\sigma^2}{1 - \phi^2}\right) \\
X_t \mid X_{t-1} = x_{t-1} &\sim \mathcal{N}\!\left(\mu + \phi(x_{t-1} - \mu), \, \sigma^2\right), \quad t \ge 1 \\
Y_t \mid X_t = x_t &\sim \mathcal{N}\!\left(0, \, \exp(x_t)\right), \quad t \ge 0
\end{aligned}$$

* **Parameter compatibility**: `StochVol` accepts both `phi` (milestone notation) and `rho` (book notation), as well as `sigma` and `sigma2`, keeping them synchronized automatically.

---

## Quick Start for Teammates

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Parallelized Pipeline
```bash
python run_pipeline.py --particles 1000 --nruns 8 --nprocs 0
```

### 3. Load Sandbox & S&P 500 Datasets in Your Module or Notebook
```python
from src.base_model import (
    StochVol,
    TRUE_SIM_PARAMS,
    load_simulated_dataset,
    load_sp500_dataset,
    run_baseline_bootstrap,
    run_parallel_smc,
)

# 1. Synthetic Sandbox Dataset (T=500, mu=-1.0, phi=0.95, sigma=0.2)
sim = load_simulated_dataset()
x_true, y_sim = sim["x_true"], sim["y_obs"]

# 2. Cleaned S&P 500 Log-Returns Dataset (T=504 trading days, 2019–2021)
sp500 = load_sp500_dataset()
dates, prices, y_sp500 = sp500["dates"], sp500["prices"], sp500["y_obs"]
```

---

## Role-Based Milestones & Module Guide

| Role | Module | M1 (Nov 1): Setup & Math | M2 (Nov 22): Sandbox Validation | M3 (Dec 6): Real S&P 500 Run |
| :--- | :--- | :--- | :--- | :--- |
| **1. Data & Base Setup** (Gabriel) | `src/base_model.py` | Base `StochVol` class & repo structure | Upload `simulated_sv_data.csv` & verify Bootstrap PF | Upload `sp500_log_returns.csv` & parallelize `run_pipeline.py` |
| **2. Live Tracking / Filtering** (Volunteer 1) | `src/filtering.py` | Taylor expansion math for Guided & APF proposals (`GuidedStochVol`) | Plot + ESS/MSE table comparing Guided/APF vs. Bootstrap on sandbox data | Final volatility tracking plots & ESS table on S&P 500 data |
| **3. Hindsight Smoothing** (Volunteer 2) | `src/smoothing.py` | Configure FFBS (`ON2`, `reject`, `mcmc`) & two-filter smoothing | Plot smoothed vs. filtered trajectory on sandbox data ($O(N^2)$ vs. fast FFBS) | S&P 500 crash turbulence reconstruction plot |
| **4. Parameter Estimation** (Volunteer 3) | `src/estimation.py` | Define `StructDist` priors for $\theta = (\mu, \phi, \sigma)$ | Posterior histograms/traces from PMCMC & $\text{SMC}^2$ recovering `TRUE_SIM_PARAMS` | Final posterior distributions & parameter estimates for S&P 500 |

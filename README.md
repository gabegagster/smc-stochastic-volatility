# SMC Cholesky Multivariate Stochastic Volatility (ChMSV) Project

This repository contains the codebase for our group project on **Multivariate Stochastic Volatility (ChMSV)** using the [`particles`](https://github.com/nchopin/particles) package in Python, based on:

> **Szczepocki, P. (2023).** *Estimation of the Cholesky Multivariate Stochastic Volatility Model Using Iterated Filtering.* Econometrics. Ekonometria. Advances in Applied Data Analysis, 27(4), 44–58.

---

## Project Structure

```text
├── data/
│   ├── raw/
│   │   ├── four_assets_raw_prices.csv       <- Aligned daily closing prices (SPX, Gold, BTC, ETH: 2018–2023)
│   │   └── stooq/                           <- Cleaned individual asset series
│   └── processed/
│       ├── real_4assets_log_returns.csv     <- Cleaned 4-asset dataset (T = 1258, 100 * daily log-returns)
│       ├── real_2assets_log_returns.csv     <- Fast 2-asset real subset (SPX & Gold, T = 1258)
│       ├── simulated_4assets_chmsv.csv      <- Full 4-asset simulated dataset (10 states, 30 params, T = 1258)
│       ├── simulated_4assets_metadata.json  <- Ground-truth Table 2 parameters & metadata (p = 4)
│       ├── simulated_2assets_chmsv.csv      <- Fast 2-asset simulated dataset (3 states, 9 params, T = 500)
│       └── simulated_2assets_metadata.json  <- Ground-truth Table 2 parameters & metadata (p = 2)
├── notebooks/
│   └── smc_stochastic_volatility.ipynb      <- Shared main Jupyter notebook (Role 1 setup + Role 2/3/4 sections)
├── plots/
│   ├── fig1_real_4assets_prices_and_returns.png <- Reproduction of Fig. 1 (Szczepocki, 2023)
│   └── simulated_chmsv_datasets_overview.png    <- Ground-truth volatilities & correlations of simulated data
├── tables/
│   ├── table1_summary_statistics.tex        <- Standalone compilable LaTeX document with descriptive & comparison tables
│   ├── table1_summary_statistics.pdf        <- Compiled PDF ready for preview / reports
│   ├── tab_descriptive_stats.tex            <- Table fragment ready to \input into reports
│   └── tab_comparison_paper.tex             <- Comparison fragment ready to \input into reports
├── src/
│   ├── __init__.py                          <- Package exports for ChMSV models, helpers, and data loaders
│   ├── base_model.py                        <- Role 1: Data & Base ChMSV Model Setup (Gabriel)
│   ├── filtering.py                         <- Placeholder for Role 2 (Bootstrap PF & Dynamic Correlations)
│   ├── smoothing.py                         <- Placeholder for teammate modules
│   └── estimation.py                        <- Placeholder for Role 3 (IF2) / Role 4 (PMCMC / SMC^2)
├── run_pipeline.py                          <- Role 1 setup & verification runner script
├── requirements.txt                         <- Python dependencies
└── README.md                                <- Project overview & teammate quick-start guide
```

---

## ChMSV Mathematical Specification (`src/base_model.py`)

For a p-dimensional vector of daily percentage log-returns y_t = (y_{1,t}, ..., y_{p,t})' (Equations 1–9 of Szczepocki, 2023):

$$
y_t = L_t V_t^{1/2} e_t, \quad e_t \sim \text{iid } \mathcal{N}_p(0, I_p)
$$

The latent states follow independent Gaussian AR(1) processes:

$$
h_{i,t} = \mu_i + \phi_i (h_{i,t-1} - \mu_i) + \eta_{i,t}, \quad \eta_{i,t} \sim \text{iid } \mathcal{N}(0, \sigma_{\eta_i}^2), \quad i = 1, \dots, p
$$

$$
q_{ij,t} = \mu_{ij} + \phi_{ij} (q_{ij,t-1} - \mu_{ij}) + \upsilon_{ij,t}, \quad \upsilon_{ij,t} \sim \text{iid } \mathcal{N}(0, \sigma_{\psi_{ij}}^2), \quad 1 \le j < i \le p
$$

### Latent State Vector (d_x = p(p + 1) / 2)

* **4-Asset Model** (`ChMSV` / `ChMSV4Asset`, p = 4, d_x = 10 hidden states, 30 parameters):
  ```text
  X_t = (h_{1,t}, h_{2,t}, h_{3,t}, h_{4,t}, q_{21,t}, q_{31,t}, q_{41,t}, q_{32,t}, q_{42,t}, q_{43,t})'
  ```

* **Fast 2-Asset Model** (`ChMSV2Asset`, p = 2, d_x = 3 hidden states, 9 parameters — SPX & Gold):
  ```text
  X_t = (h_{1,t}, h_{2,t}, q_{21,t})'
  ```

### State-to-Correlation Transformations (Equations 4, 5, 6, 7)

The conditional covariance matrix Sigma_t = L_t V_t L_t' has diagonal elements (q_{ii,t} = 1):

$$
\Sigma_{t,ii} = \sum_{k=1}^i q_{ik,t}^2 \exp(h_{k,t})
$$

and off-diagonal elements (i > j):

$$
\Sigma_{t,ij} = \sum_{k=1}^j q_{ik,t} q_{jk,t} \exp(h_{k,t})
$$

The conditional standard deviations (volatilities) and dynamic correlations are extracted as:

$$
\text{Stdev}(y_{i,t} \mid \mathcal{F}_{t-1}) = \sqrt{\Sigma_{t,ii}}
$$

$$
\text{Cor}(y_{i,t}, y_{j,t} \mid \mathcal{F}_{t-1}) = \frac{\Sigma_{t,ij}}{\sqrt{\Sigma_{t,ii} \, \Sigma_{t,jj}}}
$$

---

## Quick Start for Teammates

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

### 2. Loading Datasets & Using `ChMSV` in Your Code
```python
from particles import state_space_models as ssm
from src.base_model import (
    ChMSV,
    ChMSV4Asset,
    ChMSV2Asset,
    TABLE2_PARAMS_4ASSET,
    TABLE2_PARAMS_2ASSET,
    load_real_dataset,
    load_simulated_chmsv_dataset,
    states_to_volatility_and_correlation,
)

# 1. Load Real 4-Asset Dataset (SPX, Gold, Bitcoin, Ethereum; T = 1258)
real4 = load_real_dataset(p=4)
y_real4 = real4["y_obs"]  # shape: (1258, 4)

# 2. Load Simulated Datasets (Fast 2-Asset for debugging & Full 4-Asset)
sim2 = load_simulated_chmsv_dataset(p=2)  # y_obs: (500, 2), x_true: (500, 3)
sim4 = load_simulated_chmsv_dataset(p=4)  # y_obs: (1258, 4), x_true: (1258, 10)

# 3. Instantiate ChMSV StateSpaceModel (defaults to Table 2 parameters)
model_4 = ChMSV4Asset()  # or ChMSV(p=4, **custom_params)
model_2 = ChMSV2Asset()  # or ChMSV(p=2, **custom_params)

# 4. Transform any state array X_t of shape (..., d_x) into Volatilities & Dynamic Correlations (Eqs. 6 & 7)
out = states_to_volatility_and_correlation(sim4["x_true"], p=4)
stdev = out["stdev"]            # shape: (1258, 4) -> Stdev(y_{i,t} | F_{t-1})
corr_pairs = out["corr_pairs"]  # shape: (1258, 6) -> (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
```

---

## Real Dataset Overview

The empirical dataset covers daily observations from **4 June 2018 to 4 June 2023** across four assets (T = 1258 common daily return observations):
* **SPX**: Standard & Poor's 500 index
* **Gold**: Spot gold (XAU/USD)
* **Bitcoin**: Spot Bitcoin (BTC/USD)
* **Ethereum**: Spot Ethereum (ETH/USD)

Following the paper's methodology, only common trading days present across all four series are included (1,259 price observations yielding T = 1,258 daily percentage log-returns, y_{i,t} = 100 * ln(P_{i,t} / P_{i,t-1})).

Cleaned datasets and raw price series are stored in:
* `data/processed/real_4assets_log_returns.csv` (4-asset percentage log-returns, T = 1258)
* `data/processed/real_2assets_log_returns.csv` (2-asset fast subset: SPX and Gold, T = 1258)
* `data/raw/four_assets_raw_prices.csv` (aligned daily closing prices)

Detailed LaTeX summary statistics and replication tables are available in the [`tables/`](tables/) directory (`tables/table1_summary_statistics.tex` and `tables/table1_summary_statistics.pdf`).

---

## Project Roles & Milestones

* **Milestone 1 (Setup & Algorithm Code):** by Nov 8 (Oct 27 for Role 1)
* **Milestone 2 (Simulated Data Benchmark & 10-Run Variability):** by Nov 29
* **Milestone 3 (Real 4-Asset Run & Final Plots):** by Dec 20 (Final submission: Jan 10)

| Role | Focus Area | Key Deliverables |
| :--- | :--- | :--- |
| **1. Data & Base ChMSV Model Setup** (Gabriel) | `src/base_model.py`, `data/`, `notebooks/`, `tables/` | 4-asset & 2-asset `ChMSV` classes in `particles`, state-to-correlation helpers (Eqs. 4–7), cleaned real 4-asset dataset (2018–2023), 2-asset & 4-asset simulated datasets, LaTeX tables |
| **2. Task 1 – Bootstrap PF & Dynamic Correlations** | `src/filtering.py` | Bootstrap PF pipeline extracting volatilities, dynamic correlations & log-likelihoods; 10-run variability benchmark on simulated data; reproduction of Figs. 2, 3, 4 on real data |
| **3. Task 2 & 3 – Iterated Filtering (IF2)** | `src/estimation.py` / `src/smoothing.py` | IF2 algorithm (Ionides et al., 2015) with random-walk parameter perturbations & geometric cooling (alpha = 0.5); 10+ run ME/RMSE recovery (Table 3) & real 4-asset estimation (Table 2) |
| **4. Task 4 – Competing Estimator (PMCMC / SMC^2)** | `src/estimation.py` | Bayesian parameter estimation via `particles.mcmc` (PMMH / Particle Gibbs) or `particles.smc_samplers` (SMC^2); multi-chain variability & head-to-head comparison vs. IF2 |

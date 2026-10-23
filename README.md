# SMC Cholesky Multivariate Stochastic Volatility (ChMSV) Project

This repository contains the codebase for our group project on **Multivariate Stochastic Volatility (ChMSV)** using the [`particles`](https://github.com/nchopin/particles) package in Python, based on:

> **Szczepocki, P. (2023).** *Estimation of the Cholesky Multivariate Stochastic Volatility Model Using Iterated Filtering.* Econometrics. Ekonometria. Advances in Applied Data Analysis, 27(4), 44–58.

---

## Project Structure

```text
├── data/
│   ├── raw/
│   │   └── four_assets_raw_prices.csv       <- Raw daily closing prices (SPX, Gold, BTC, ETH: 2018–2023)
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

For a $p$-dimensional vector of daily percentage log-returns $\mathbb{y}_t = (y_{1,t}, \dots, y_{p,t})'$ (Equations 1–9 of Szczepocki, 2023):

$$\begin{aligned}
\mathbb{y}_t &= \mathbb{L}_t \mathbb{V}_t^{1/2} \mathbb{e}_t, \quad \mathbb{e}_t \sim \text{iid } \mathcal{N}_p(0, \mathbb{I}_p), \\
h_{i,t} &= \mu_i + \phi_i (h_{i,t-1} - \mu_i) + \eta_{i,t}, \quad \eta_{i,t} \sim \text{iid } \mathcal{N}(0, \sigma_{\eta_i}^2), \quad i = 1, \dots, p, \\
q_{ij,t} &= \mu_{ij} + \phi_{ij} (q_{ij,t-1} - \mu_{ij}) + \upsilon_{ij,t}, \quad \upsilon_{ij,t} \sim \text{iid } \mathcal{N}(0, \sigma_{\psi_{ij}}^2), \quad 1 \le j < i \le p.
\end{aligned}$$

* **Hidden State Vector ($d_x = p(p+1)/2$):**
  * **4-Asset (`ChMSV` / `ChMSV4Asset`, $p=4$, $d_x=10$, 30 parameters):**
    $$X_t = \left(h_{1,t}, h_{2,t}, h_{3,t}, h_{4,t}, \, q_{21,t}, q_{31,t}, q_{41,t}, q_{32,t}, q_{42,t}, q_{43,t}\right)'$$
  * **Fast 2-Asset (`ChMSV2Asset`, $p=2$, $d_x=3$, 9 parameters — SPX & Gold):**
    $$X_t = \left(h_{1,t}, h_{2,t}, \, q_{21,t}\right)'$$
* **State-to-Correlation Transformations (Equations 4, 5, 6, 7):**
  $$\Sigma_{t,ii} = \sum_{k=1}^i q_{ik,t}^2 \exp(h_{k,t}), \qquad \Sigma_{t,ij} = \sum_{k=1}^j q_{ik,t} q_{jk,t} \exp(h_{k,t}) \quad (i > j),$$
  $$\text{Stdev}(y_{i,t} \mid \mathcal{F}_{t-1}) = \sqrt{\Sigma_{t,ii}}, \qquad \text{Cor}(y_{i,t}, y_{j,t} \mid \mathcal{F}_{t-1}) = \frac{\Sigma_{t,ij}}{\sqrt{\Sigma_{t,ii}\Sigma_{t,jj}}}.$$

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

## Real Data: Source & Benchmark against Table 1 (Szczepocki, 2023)

The dataset covers **4 June 2018 to 4 June 2023** ($T = 1258$ common daily return observations):
1. **SPX (`^spx`)**: Standard and Poor's 500 index. Matches Table 1 across every single statistic (`Mean=0.035, Std=1.374, Skew=-0.770, Kurt=16.221, Min=-12.765, Q1=-0.533, Median=0.077, Q3=0.712, Max=8.968`).
2. **Gold (`xauusd`)**: Spot XAU/USD price retrieved directly from `stooq.pl`. Matches Table 1 exactly (`Mean=0.033, Std=0.926, Skew=-0.413, Kurt=6.528, Min=-5.868, Q1=-0.432, Median=0.070, Q3=0.541, Max=4.605`).
3. **Bitcoin (`btcusd`)**: Spot BTC/USD currency pair retrieved from `stooq.pl`. Matches all 5 Table 1 quantiles to the decimal place (`Min=-32.360, Q1=-1.757, Median=0.000, Q3=2.059, Max=23.230`), and moment statistics (`Std=4.302 vs 4.305, Skew=-0.262 vs -0.260, Kurt=8.485 vs 8.460`).
4. **Ethereum (`eth.v`)**: Spot ETH/USD aligned to Warsaw midnight (`00:00 Europe/Warsaw`, Stooq's close convention). Matches Table 1 closely (`Min=-42.446 vs -42.855, Std=5.693 vs 5.715, Skew=-0.445 vs -0.442, Kurt=8.553 vs 8.594, Q1=-2.418 vs -2.408, Q3=2.886 vs 2.887, Max=31.151 vs 31.419`), eliminating the massive divergence seen in standard UTC-midnight Yahoo data (which had `Min=-55.073` and `Kurt=12.619`).
5. **Log-Likelihood Benchmark**: Evaluating the Bootstrap Particle Filter ($N = 1000$, 12 independent runs) at Table 2 parameters yields a mean log-likelihood of **$-9977.26$** (within 2 standard errors of the paper's reported $-9972.73 \pm 0.65$).

Clean series are stored in `data/raw/four_assets_raw_prices.csv` and `data/raw/stooq/`. Run `python run_pipeline.py` to regenerate all processed datasets and verification plots.

---

## Project Roles & Milestones

* **Milestone 1 (Setup & Algorithm Code):** by Nov 8 (Oct 27 for Role 1)
* **Milestone 2 (Simulated Data Benchmark & 10-Run Variability):** by Nov 29
* **Milestone 3 (Real 4-Asset Run & Final Plots):** by Dec 20 (Final submission: Jan 10)

| Role | Focus Area | Key Deliverables |
| :--- | :--- | :--- |
| **1. Data & Base ChMSV Model Setup** (Gabriel) | `src/base_model.py`, `data/`, `notebooks/` | 4-asset & 2-asset `ChMSV` classes in `particles`, state-to-correlation helpers (Eqs. 4–7), cleaned real 4-asset dataset (2018–2023), 2-asset & 4-asset simulated datasets |
| **2. Task 1 – Bootstrap PF & Dynamic Correlations** | `src/filtering.py` | Bootstrap PF pipeline extracting volatilities, dynamic correlations & log-likelihoods; 10-run variability benchmark on simulated data; reproduction of Figs. 2, 3, 4 on real data |
| **3. Task 2 & 3 – Iterated Filtering (IF2)** | `src/estimation.py` / `src/smoothing.py` | IF2 algorithm (Ionides et al., 2015) with random-walk parameter perturbations & geometric cooling ($\alpha = 0.5$); 10+ run ME/RMSE recovery (Table 3) & real 4-asset estimation (Table 2) |
| **4. Task 4 – Competing Estimator (PMCMC / $\text{SMC}^2$)** | `src/estimation.py` | Bayesian parameter estimation via `particles.mcmc` (PMMH / Particle Gibbs) or `particles.smc_samplers` ($\text{SMC}^2$); multi-chain variability & head-to-head comparison vs. IF2 |

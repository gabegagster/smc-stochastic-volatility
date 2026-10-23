"""
Role 1: Data & Base ChMSV Model Setup (Gabriel)

Implements the Cholesky Multivariate Stochastic Volatility (ChMSV) state-space model
and data pipeline based on:
    Szczepocki, P. (2023). "Estimation of the Cholesky Multivariate Stochastic
    Volatility Model Using Iterated Filtering." Econometrics. Ekonometria.
    Advances in Applied Data Analysis, 27(4), 44–58.

Deliverables in this module:
1. `ChMSV` (`ChMSV4Asset`, `ChMSV2Asset`) subclassing `particles.state_space_models.StateSpaceModel`:
   - Supports general p-dimensional ChMSV models:
     * Full 4-asset model (p = 4): 10 hidden states, 30 parameters (Table 2 defaults)
     * Fast 2-asset model (p = 2): 3 hidden states, 9 parameters (SPX & Gold subset)
   - Vectorized, zero-overhead custom `ProbDist` classes (`ChMSVInitDist`,
     `ChMSVTransDist`, `ChMSVObsDist`) implementing Equations (1)–(3) & (8)–(11).
2. State-to-Correlation transformation helpers (Equations 4, 5, 6, 7):
   - `states_to_cholesky_factors`
   - `states_to_covariance`
   - `states_to_volatility_and_correlation`
3. Real 4-asset dataset pipeline (SPX, Gold, Bitcoin, Ethereum, June 4, 2018 – June 4, 2023,
   T = 1258 daily percentage log-returns) and Table 1 summary statistics verification.
4. Simulated test dataset generators and loaders:
   - Full 4-asset simulated dataset (`data/processed/simulated_4assets_chmsv.csv`, T = 1258)
   - Fast 2-asset simulated dataset (`data/processed/simulated_2assets_chmsv.csv`, T = 500)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from particles import distributions as dists
from particles import state_space_models as ssm
from scipy import stats

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "plots"

RAW_4ASSETS_CSV_PATH = DATA_RAW_DIR / "four_assets_raw_prices.csv"
REAL_4ASSETS_CSV_PATH = DATA_PROCESSED_DIR / "real_4assets_log_returns.csv"
REAL_2ASSETS_CSV_PATH = DATA_PROCESSED_DIR / "real_2assets_log_returns.csv"

SIM_4ASSETS_CSV_PATH = DATA_PROCESSED_DIR / "simulated_4assets_chmsv.csv"
SIM_4ASSETS_META_PATH = DATA_PROCESSED_DIR / "simulated_4assets_metadata.json"
SIM_2ASSETS_CSV_PATH = DATA_PROCESSED_DIR / "simulated_2assets_chmsv.csv"
SIM_2ASSETS_META_PATH = DATA_PROCESSED_DIR / "simulated_2assets_metadata.json"

ASSET_NAMES_4: Tuple[str, ...] = ("SPX", "Gold", "Bitcoin", "Ethereum")
ASSET_NAMES_2: Tuple[str, ...] = ("SPX", "Gold")


# ---------------------------------------------------------------------------
# Indexing Conventions for ChMSV Latent States & Parameters
# ---------------------------------------------------------------------------
def get_offdiag_pairs(p: int) -> List[Tuple[int, int]]:
    """Return off-diagonal index pairs (i, j) with i > j (1-based) in Table 2 order.

    In Szczepocki (2023, Table 2 & Table 3), the (p-1)p/2 lower-triangular entries
    q_{ij,t} (row i > col j of L_t) are ordered by column j = 1, ..., p-1 and
    row i = j+1, ..., p:
      - For p = 2: [(2, 1)]                         -> pair (1,2) in Table 2
      - For p = 4: [(2, 1), (3, 1), (4, 1),
                    (3, 2), (4, 2), (4, 3)]         -> pairs (1,2), (1,3), (1,4),
                                                             (2,3), (2,4), (3,4)
    """
    if p < 2:
        raise ValueError(f"ChMSV requires p >= 2 assets, got p={p}.")
    pairs: List[Tuple[int, int]] = []
    for j in range(1, p):
        for i in range(j + 1, p + 1):
            pairs.append((i, j))
    return pairs


def get_state_names(p: int) -> List[str]:
    """Return ordered list of latent state names X_t = (h_1..h_p, q_21..q_{p,p-1})."""
    h_names = [f"h_{i}" for i in range(1, p + 1)]
    q_names = [f"q_{i}{j}" for (i, j) in get_offdiag_pairs(p)]
    return h_names + q_names


# ---------------------------------------------------------------------------
# Table 2 Parameters from Szczepocki (2023, p. 52)
# ---------------------------------------------------------------------------
# Exact IF2 parameter estimates reported in Table 2 for the 4-asset empirical study:
# SPX (i=1), Gold (i=2), Bitcoin (i=3), Ethereum (i=4)
TABLE2_PARAMS_4ASSET: Dict[str, float] = {
    # Log-volatility processes h_{i,t} (i = 1, 2, 3, 4)
    "mu_1": 0.0012,
    "phi_1": 0.9549,
    "sigma_1": 0.1357,
    "mu_2": -0.1888,
    "phi_2": 0.8545,
    "sigma_2": 0.4811,
    "mu_3": 2.1081,
    "phi_3": 0.7445,
    "sigma_3": 0.9101,
    "mu_4": 1.3398,
    "phi_4": 0.8650,
    "sigma_4": 0.5441,
    # Dynamic correlation Cholesky processes q_{ij,t} (i > j)
    # Pair (1,2) -> q_{21,t} (SPX, Gold)
    "mu_21": 0.0472,
    "phi_21": 0.5710,
    "sigma_21": 0.0022,
    # Pair (1,3) -> q_{31,t} (SPX, Bitcoin)
    "mu_31": 0.1403,
    "phi_31": 0.3778,
    "sigma_31": 0.0519,
    # Pair (1,4) -> q_{41,t} (SPX, Ethereum)
    "mu_41": 0.1267,
    "phi_41": 0.8327,
    "sigma_41": 0.0067,
    # Pair (2,3) -> q_{32,t} (Gold, Bitcoin)
    "mu_32": 0.1447,
    "phi_32": 0.9658,
    "sigma_32": 0.0904,
    # Pair (2,4) -> q_{42,t} (Gold, Ethereum)
    "mu_42": 0.1755,
    "phi_42": 0.9904,
    "sigma_42": 0.1081,
    # Pair (3,4) -> q_{43,t} (Bitcoin, Ethereum)
    "mu_43": 1.0006,
    "phi_43": 0.4927,
    "sigma_43": 0.3268,
}

# Standard errors reported in brackets in Table 2 (Szczepocki, 2023, p. 52)
TABLE2_STD_ERRORS_4ASSET: Dict[str, float] = {
    "mu_1": 0.0409,
    "phi_1": 0.0040,
    "sigma_1": 0.0042,
    "mu_2": 0.0446,
    "phi_2": 0.0049,
    "sigma_2": 0.0051,
    "mu_3": 0.0533,
    "phi_3": 0.0050,
    "sigma_3": 0.0053,
    "mu_4": 0.0436,
    "phi_4": 0.0044,
    "sigma_4": 0.0047,
    "mu_21": 0.0197,
    "phi_21": 0.0049,
    "sigma_21": 0.0006,
    "mu_31": 0.0364,
    "phi_31": 0.0047,
    "sigma_31": 0.0005,
    "mu_41": 0.0450,
    "phi_41": 0.0051,
    "sigma_41": 0.0005,
    "mu_32": 0.0407,
    "phi_32": 0.0049,
    "sigma_32": 0.0006,
    "mu_42": 0.0470,
    "phi_42": 0.0040,
    "sigma_42": 0.0005,
    "mu_43": 0.0265,
    "phi_43": 0.0048,
    "sigma_43": 0.0005,
}

TABLE2_LOG_LIKELIHOOD = -9972.7344
TABLE2_LOG_LIKELIHOOD_SE = 0.6468

# 2-Asset subset (SPX & Gold, p = 2, 3 states, 9 parameters) for fast debugging
TABLE2_PARAMS_2ASSET: Dict[str, float] = {
    "mu_1": TABLE2_PARAMS_4ASSET["mu_1"],
    "phi_1": TABLE2_PARAMS_4ASSET["phi_1"],
    "sigma_1": TABLE2_PARAMS_4ASSET["sigma_1"],
    "mu_2": TABLE2_PARAMS_4ASSET["mu_2"],
    "phi_2": TABLE2_PARAMS_4ASSET["phi_2"],
    "sigma_2": TABLE2_PARAMS_4ASSET["sigma_2"],
    "mu_21": TABLE2_PARAMS_4ASSET["mu_21"],
    "phi_21": TABLE2_PARAMS_4ASSET["phi_21"],
    "sigma_21": TABLE2_PARAMS_4ASSET["sigma_21"],
}


def get_parameter_names(p: int = 4) -> List[str]:
    """Return the 3*p(p+1)/2 canonical parameter names in order."""
    names: List[str] = []
    for i in range(1, p + 1):
        names.extend([f"mu_{i}", f"phi_{i}", f"sigma_{i}"])
    for i, j in get_offdiag_pairs(p):
        names.extend([f"mu_{i}{j}", f"phi_{i}{j}", f"sigma_{i}{j}"])
    return names


def params_dict_to_vectors(
    params: Dict[str, Any], p: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a dictionary of ChMSV parameters into (mu_vec, phi_vec, sigma_vec) of length d_x.

    Supports multiple naming conventions for teammate convenience:
    - Volatility noise std: `sigma_i` or `sigma_eta_i` (or variance `sigma2_i`)
    - Cholesky off-diagonal parameters: `mu_ij` (i > j, e.g. `mu_21`) OR Table 2 pair
      notation `mu_ji` (j < i, e.g. `mu_12`), and `sigma_ij` / `sigma_ji` / `sigma_psi_ij`.
    """
    pairs = get_offdiag_pairs(p)
    dx = p * (p + 1) // 2

    if "mu" in params and isinstance(params["mu"], (list, tuple, np.ndarray)):
        mu_vec = np.asarray(params["mu"], dtype=float).ravel()
        phi_vec = np.asarray(params["phi"], dtype=float).ravel()
        sigma_vec = np.asarray(params["sigma"], dtype=float).ravel()
        if len(mu_vec) == dx and len(phi_vec) == dx and len(sigma_vec) == dx:
            return mu_vec.copy(), phi_vec.copy(), np.maximum(sigma_vec, 1e-8)

    mu_list: List[float] = []
    phi_list: List[float] = []
    sigma_list: List[float] = []

    defaults = TABLE2_PARAMS_4ASSET if p == 4 else TABLE2_PARAMS_2ASSET

    for i in range(1, p + 1):
        mu_i = params.get(f"mu_{i}", defaults.get(f"mu_{i}", 0.0))
        phi_i = params.get(f"phi_{i}", defaults.get(f"phi_{i}", 0.9))
        if f"sigma_{i}" in params:
            sig_i = params[f"sigma_{i}"]
        elif f"sigma_eta_{i}" in params:
            sig_i = params[f"sigma_eta_{i}"]
        elif f"sigma2_{i}" in params:
            sig_i = np.sqrt(max(float(params[f"sigma2_{i}"]), 1e-12))
        else:
            sig_i = defaults.get(f"sigma_{i}", 0.2)
        mu_list.append(float(mu_i))
        phi_list.append(float(phi_i))
        sigma_list.append(max(float(sig_i), 1e-8))

    for i, j in pairs:
        # Support both (i,j) with i>j (e.g. 21) and Table 2 pair (j,i) with j<i (e.g. 12)
        ij, ji = f"{i}{j}", f"{j}{i}"
        mu_ij = params.get(
            f"mu_{ij}", params.get(f"mu_{ji}", defaults.get(f"mu_{ij}", 0.0))
        )
        phi_ij = params.get(
            f"phi_{ij}", params.get(f"phi_{ji}", defaults.get(f"phi_{ij}", 0.5))
        )
        sig_ij = params.get(
            f"sigma_{ij}",
            params.get(
                f"sigma_{ji}",
                params.get(
                    f"sigma_psi_{ij}",
                    params.get(
                        f"sigma_psi_{ji}",
                        defaults.get(f"sigma_{ij}", 0.05),
                    ),
                ),
            ),
        )
        mu_list.append(float(mu_ij))
        phi_list.append(float(phi_ij))
        sigma_list.append(max(float(sig_ij), 1e-8))

    return (
        np.asarray(mu_list, dtype=float),
        np.asarray(phi_list, dtype=float),
        np.asarray(sigma_list, dtype=float),
    )


def vectors_to_params_dict(
    mu_vec: np.ndarray,
    phi_vec: np.ndarray,
    sigma_vec: np.ndarray,
    p: int = 4,
) -> Dict[str, float]:
    """Convert (mu_vec, phi_vec, sigma_vec) arrays of length d_x back to a named parameter dict."""
    pairs = get_offdiag_pairs(p)
    out: Dict[str, float] = {}
    for idx in range(p):
        i = idx + 1
        out[f"mu_{i}"] = float(mu_vec[idx])
        out[f"phi_{i}"] = float(phi_vec[idx])
        out[f"sigma_{i}"] = float(sigma_vec[idx])
    for k, (i, j) in enumerate(pairs):
        idx = p + k
        out[f"mu_{i}{j}"] = float(mu_vec[idx])
        out[f"phi_{i}{j}"] = float(phi_vec[idx])
        out[f"sigma_{i}{j}"] = float(sigma_vec[idx])
    return out


# ---------------------------------------------------------------------------
# 1. Helper Functions: Transform Hidden States -> Volatilities & Correlations
#    (Equations 4, 5, 6, and 7 in Szczepocki, 2023, p. 47)
# ---------------------------------------------------------------------------
def states_to_cholesky_factors(
    x: np.ndarray, p: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract unit lower-triangular matrix L_t and diagonal variance entries exp(h_t).

    Parameters
    ----------
    x : np.ndarray
        Latent state array of shape `(..., d_x)` where `d_x = p * (p + 1) // 2`.
        Works for a single state `(d_x,)`, a particle cloud `(N, d_x)`,
        a state trajectory `(T, d_x)`, or a 3D history `(T, N, d_x)`.
    p : int
        Number of assets (default: 4).

    Returns
    -------
    L : np.ndarray
        Unit lower-triangular matrices of shape `(..., p, p)`.
    v_diag : np.ndarray
        Diagonal variance elements `exp(h_{i,t})` of shape `(..., p)`.
    """
    x_arr = np.asarray(x, dtype=float)
    dx = p * (p + 1) // 2
    if x_arr.shape[-1] != dx:
        raise ValueError(
            f"Expected last dimension d_x={dx} for p={p} assets, got shape {x_arr.shape}."
        )

    batch_shape = x_arr.shape[:-1]
    h = x_arr[..., :p]
    q = x_arr[..., p:]

    v_diag = np.exp(h)
    L = np.zeros(batch_shape + (p, p), dtype=float)
    for idx in range(p):
        L[..., idx, idx] = 1.0

    for k, (i, j) in enumerate(get_offdiag_pairs(p)):
        L[..., i - 1, j - 1] = q[..., k]

    return L, v_diag


def states_to_covariance(x: np.ndarray, p: int = 4) -> np.ndarray:
    """Compute time-varying conditional covariance matrix Sigma_t = L_t V_t L_t' (Eqs. 4 & 5).

    Specifically:
    - Diagonal elements (Eq. 4):
      Sigma_{t,ii} = sum_{k=1}^i q_{ik,t}^2 exp(h_{k,t}),  q_{ii,t} = 1
    - Off-diagonal elements (Eq. 5):
      Sigma_{t,ij} = sum_{k=1}^j q_{ik,t} q_{jk,t} exp(h_{k,t}),  i > j

    Parameters
    ----------
    x : np.ndarray
        Latent state array of shape `(..., d_x)`.
    p : int
        Number of assets (default: 4).

    Returns
    -------
    Sigma : np.ndarray
        Conditional covariance matrix of shape `(..., p, p)`.
    """
    L, v_diag = states_to_cholesky_factors(x, p=p)
    # Sigma_{..., i, j} = sum_k L_{..., i, k} * v_diag_{..., k} * L_{..., j, k}
    Lv = L * v_diag[..., None, :]
    Sigma = np.einsum("...ik,...jk->...ij", Lv, L)
    return Sigma


def states_to_volatility_and_correlation(
    x: np.ndarray, p: int = 4
) -> Dict[str, np.ndarray]:
    """Transform hidden states X_t into return standard deviations (Eq. 6) and dynamic correlations (Eq. 7).

    Formulas (Szczepocki, 2023, p. 47):
    - Equation (6): Stdev(y_{i,t} | F_{t-1}) = sqrt(Sigma_{t,ii})
    - Equation (7): Cor(y_{i,t}, y_{j,t} | F_{t-1}) = Sigma_{t,ij} / sqrt(Sigma_{t,ii} * Sigma_{t,jj})

    Parameters
    ----------
    x : np.ndarray
        Latent state array of shape `(..., d_x)`.
    p : int
        Number of assets (default: 4).

    Returns
    -------
    dict
        Dictionary with keys:
        - `'stdev'`: Return standard deviations sqrt(Sigma_{t,ii}) of shape `(..., p)` (Eq. 6)
        - `'variance'`: Return variances Sigma_{t,ii} of shape `(..., p)` (Eq. 4)
        - `'cov_matrix'`: Full covariance matrices Sigma_t of shape `(..., p, p)` (Eqs. 4 & 5)
        - `'corr_matrix'`: Full correlation matrices of shape `(..., p, p)` (Eq. 7)
        - `'corr_pairs'`: Off-diagonal dynamic correlations of shape `(..., p*(p-1)//2)`
          ordered as in Table 2 & Figures 3–4:
          ((1,2), (1,3), (1,4), (2,3), (2,4), (3,4)) for p = 4, or ((1,2),) for p = 2.
    """
    Sigma = states_to_covariance(x, p=p)
    variances = np.diagonal(Sigma, axis1=-2, axis2=-1)
    variances = np.maximum(variances, 1e-16)
    stdev = np.sqrt(variances)

    denom = stdev[..., :, None] * stdev[..., None, :]
    corr_matrix = np.clip(Sigma / denom, -1.0, 1.0)
    # Enforce exact unit diagonal
    for idx in range(p):
        corr_matrix[..., idx, idx] = 1.0

    pairs = get_offdiag_pairs(p)
    corr_pairs = np.stack(
        [corr_matrix[..., i - 1, j - 1] for (i, j) in pairs], axis=-1
    )

    return {
        "stdev": stdev,
        "variance": variances,
        "cov_matrix": Sigma,
        "corr_matrix": corr_matrix,
        "corr_pairs": corr_pairs,
    }


# ---------------------------------------------------------------------------
# 2. Vectorized Probability Distributions for `particles`
# ---------------------------------------------------------------------------
class ChMSVInitDist(dists.ProbDist):
    """Point-mass (Dirac) initial distribution for X_0 at mu (Szczepocki, 2023, p. 48)."""

    def __init__(self, loc: np.ndarray) -> None:
        self.loc = np.asarray(loc, dtype=float).ravel()
        self.dim = int(self.loc.shape[0])
        self.dtype = float

    def rvs(self, size: Optional[int] = None) -> np.ndarray:
        N = 1 if size is None else int(size)
        return np.tile(self.loc, (N, 1))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            is_eq = np.allclose(x_arr, self.loc)
            return np.array(0.0 if is_eq else -np.inf)
        is_eq = np.all(np.isclose(x_arr, self.loc[None, :]), axis=-1)
        return np.where(is_eq, 0.0, -np.inf)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u, dtype=float)
        N = 1 if u_arr.ndim == 1 else int(u_arr.shape[0])
        return np.tile(self.loc, (N, 1))


class ChMSVTransDist(dists.ProbDist):
    """Diagonal multivariate Gaussian transition distribution P_t(x_{t-1}, dx_t) (Eqs. 8–11).

    Avoids O(d_x^3) Cholesky factorizations of diagonal covariance matrices at every step t.
    """

    def __init__(self, loc: np.ndarray, scale: np.ndarray) -> None:
        self.loc = np.asarray(loc, dtype=float)
        self.scale = np.asarray(scale, dtype=float).ravel()
        self.dim = int(self.scale.shape[0])
        self.dtype = float
        self._norm_cst = float(
            0.5 * self.dim * np.log(2.0 * np.pi) + np.sum(np.log(self.scale))
        )

    def rvs(self, size: Optional[int] = None) -> np.ndarray:
        if self.loc.ndim == 1:
            N = 1 if size is None else int(size)
            z = np.random.normal(size=(N, self.dim))
            return self.loc[None, :] + z * self.scale[None, :]
        N = int(self.loc.shape[0]) if size is None else int(size)
        z = np.random.normal(size=(N, self.dim))
        return self.loc + z * self.scale[None, :]

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        z = (x_arr - self.loc) / self.scale
        return -0.5 * np.sum(z * z, axis=-1) - self._norm_cst

    def ppf(self, u: np.ndarray) -> np.ndarray:
        z = stats.norm.ppf(u)
        return self.loc + z * self.scale


class ChMSVObsDist(dists.ProbDist):
    r"""Observation emission distribution y_t | X_t ~ N_p(0, L_t V_t L_t') (Eqs. 1 & 3).

    Exploits the LDL/Cholesky structure directly via forward substitution:
        L_t z_t = y_t  ==>  z_{i,t} = y_{i,t} - \sum_{j < i} q_{ij,t} z_{j,t}
    Since det(L_t) = 1 and z_{i,t} ~ indep. N(0, exp(h_{i,t})), the exact log-density is:
        log g_t(y_t | X_t) = -0.5 * p * log(2*pi) - 0.5 * \sum_{i=1}^p (h_{i,t} + z_{i,t}^2 * exp(-h_{i,t}))
    This requires zero matrix inversions and zero Cholesky decompositions across all N particles.
    """

    def __init__(self, x: np.ndarray, p: int = 4) -> None:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        self.x = x_arr
        self.N = int(x_arr.shape[0])
        self.p = int(p)
        self.dim = int(p)
        self.dtype = float
        self.pairs = get_offdiag_pairs(p)

    def rvs(self, size: Optional[int] = None) -> np.ndarray:
        """Simulate y_t = L_t V_t^{1/2} e_t with e_t ~ N_p(0, I_p) (Eq. 1)."""
        N = self.N if size is None else int(size)
        h = self.x[:, : self.p]
        q = self.x[:, self.p :]

        # Clamp h slightly only against numerical overflow in exp
        h_safe = np.clip(h, -50.0, 50.0)
        e = np.random.normal(size=(N, self.p))
        z = np.exp(0.5 * h_safe) * e

        y = z.copy()
        for k, (i, j) in enumerate(self.pairs):
            y[:, i - 1] += q[:, k] * z[:, j - 1]
        return y

    def logpdf(self, y: np.ndarray) -> np.ndarray:
        """Evaluate log g_t(y_t | X_t^n; theta) for all particles n = 1, ..., N."""
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr[None, :]
        elif y_arr.ndim == 2 and y_arr.shape[0] == 1 and self.N > 1:
            pass

        h = self.x[:, : self.p]
        q = self.x[:, self.p :]
        h_safe = np.clip(h, -50.0, 50.0)

        # Forward substitution to solve L_t z_t = y_t across all N particles in O(p^2)
        z = np.empty((self.N, self.p), dtype=float)
        z[:, :] = y_arr

        # Group off-diagonal elements by row i = 2, ..., p so z[:, j-1] is already finalized
        pair_map: Dict[Tuple[int, int], int] = {
            pair: k for k, pair in enumerate(self.pairs)
        }
        for i in range(2, self.p + 1):
            for j in range(1, i):
                k = pair_map[(i, j)]
                z[:, i - 1] -= q[:, k] * z[:, j - 1]

        quad_form = np.sum((z * z) * np.exp(-h_safe), axis=1)
        log_det = np.sum(h_safe, axis=1)
        norm_cst = 0.5 * self.p * np.log(2.0 * np.pi)
        return -norm_cst - 0.5 * (log_det + quad_form)


# ---------------------------------------------------------------------------
# 3. ChMSV StateSpaceModel Class in `particles`
# ---------------------------------------------------------------------------
class ChMSV(ssm.StateSpaceModel):
    r"""Cholesky Multivariate Stochastic Volatility (ChMSV) StateSpaceModel in `particles`.

    Implements the state-space representation of Szczepocki (2023, Section 2 & 3.1):
    - Observation dimension: `p` assets (default `p = 4`: SPX, Gold, Bitcoin, Ethereum).
    - Latent state dimension: `d_x = p * (p + 1) // 2` (`10` states for `p = 4`, `3` states for `p = 2`):
      :math:`X_t = (h_{1,t}, \dots, h_{p,t}, q_{21,t}, \dots, q_{p,p-1,t})'`
    - Parameter count: `3 * d_x = 3 * p * (p + 1) // 2` (`30` parameters for `p = 4`, `9` for `p = 2`).

    Initial state distribution (`PX0`, p. 48):
    - Point-mass (Dirac) at :math:`X_0 = (\mu_1, \dots, \mu_p, \mu_{21}, \dots, \mu_{p,p-1})'`
      by default (`init_dist='dirac'`), or stationary Gaussian (`init_dist='stationary'`).

    State transition density (`PX`, Eqs. 8–11):
    .. math::
        h_{i,t} \mid h_{i,t-1} &\sim \mathcal{N}\left(\mu_i + \phi_i(h_{i,t-1} - \mu_i), \sigma_{\eta_i}^2\right), \quad i = 1, \dots, p \\
        q_{ij,t} \mid q_{ij,t-1} &\sim \mathcal{N}\left(\mu_{ij} + \phi_{ij}(q_{ij,t-1} - \mu_{ij}), \sigma_{\psi_{ij}}^2\right), \quad 1 \le j < i \le p

    Observation emission density (`PY`, Eqs. 1, 3, 4, 5):
    .. math::
        \mathbb{y}_t \mid X_t \sim \mathcal{N}_p(0, \Sigma_t), \quad \Sigma_t = \mathbb{L}_t \mathbb{V}_t \mathbb{L}_t^T
    """

    p: int = 4
    default_params: Dict[str, Any] = dict(TABLE2_PARAMS_4ASSET)

    def __init__(self, p: Optional[int] = None, init_dist: str = "dirac", **kwargs: Any) -> None:
        if p is not None:
            self.p = int(p)
        else:
            self.p = int(getattr(self.__class__, "p", 4))

        self.init_dist = str(init_dist).lower()
        self.dx = self.p * (self.p + 1) // 2
        self.num_params = 3 * self.dx

        # Resolve parameters from the user-supplied kwargs only. `params_dict_to_vectors`
        # falls back to the Table 2 defaults for any missing parameter, so aliases such as
        # `mu_12` (Table 2 pair notation) or `sigma2_1` are not shadowed by merged defaults.
        # The base-class `__init__` is deliberately not called, since it would copy the
        # 4-asset `default_params` onto 2-asset instances.
        self.mu_vec, self.phi_vec, self.sigma_vec = params_dict_to_vectors(
            kwargs, p=self.p
        )
        # Keep scalar attributes synchronized with vector attributes
        canonical_dict = vectors_to_params_dict(
            self.mu_vec, self.phi_vec, self.sigma_vec, p=self.p
        )
        self.__dict__.update(kwargs)
        self.__dict__.update(canonical_dict)

    def sig0_vec(self) -> np.ndarray:
        """Stationary standard deviation vector sigma / sqrt(1 - phi^2)."""
        denom = np.maximum(1.0 - self.phi_vec**2, 1e-6)
        return self.sigma_vec / np.sqrt(denom)

    def EXt(self, xp: np.ndarray) -> np.ndarray:
        """Conditional mean vector u(x_{t-1}) = mu + phi * (x_{t-1} - mu) (Eq. 8)."""
        xp_arr = np.asarray(xp, dtype=float)
        return self.mu_vec + self.phi_vec * (xp_arr - self.mu_vec)

    def PX0(self) -> dists.ProbDist:
        """Initial distribution P_0(dx_0) (Dirac at mu by default as on p. 48 of paper)."""
        if self.init_dist == "stationary":
            return ChMSVTransDist(loc=self.mu_vec, scale=self.sig0_vec())
        return ChMSVInitDist(loc=self.mu_vec)

    def PX(self, t: int, xp: np.ndarray) -> dists.ProbDist:
        """Transition distribution P_t(x_{t-1}, dx_t) = N_{d_x}(u(x_{t-1}), S) (Eqs. 8–11)."""
        return ChMSVTransDist(loc=self.EXt(xp), scale=self.sigma_vec)

    def PY(
        self,
        t: int,
        xp: Optional[np.ndarray],
        x: np.ndarray,
    ) -> dists.ProbDist:
        """Measurement distribution g_t(y_t | x_t; theta) = N_p(0, Sigma_t) (Eqs. 1–5)."""
        return ChMSVObsDist(x=x, p=self.p)

    def transform_states(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Transform states X_t into stdev (Eq. 6), dynamic correlations (Eq. 7), and Sigma_t."""
        return states_to_volatility_and_correlation(x, p=self.p)


class ChMSV4Asset(ChMSV):
    """Explicit 4-Asset ChMSV class (10 hidden states, 30 parameters) for PMMH / SMC^2.

    Pass `ChMSV4Asset` / `ChMSV2Asset` (not the generic `ChMSV`) as `ssm_cls` to
    `particles.mcmc` / `particles.smc_samplers`: classmethods such as `state_container`
    (used by Particle Gibbs) instantiate `cls()` without arguments, so the dimension must
    be fixed at the class level.
    """

    p: int = 4
    default_params: Dict[str, Any] = dict(TABLE2_PARAMS_4ASSET)


class ChMSV2Asset(ChMSV):
    """Explicit 2-Asset ChMSV class (3 hidden states, 9 parameters: SPX & Gold) for fast debugging."""

    p: int = 2
    default_params: Dict[str, Any] = dict(TABLE2_PARAMS_2ASSET)


# ---------------------------------------------------------------------------
# 4. Real 4-Asset Dataset Pipeline (Szczepocki, 2023, Section 4, pp. 50–51)
#    Assets: SPX, Gold, Bitcoin, Ethereum (June 4, 2018 – June 4, 2023, T = 1258)
# ---------------------------------------------------------------------------
STOOQ_DIR = DATA_RAW_DIR / "stooq"
# Stooq symbols used by Szczepocki (2023, Sec. 4: "The data came from stooq.pl database").
# Download each manually from https://stooq.pl/q/d/?s=<symbol> (daily, CSV) and save as
# data/raw/stooq/<symbol>_d.csv (Stooq blocks scripted downloads with a JS challenge).
STOOQ_SYMBOLS: Dict[str, str] = {
    "SPX": "^spx",
    "Gold": "xauusd",
    "Bitcoin": "btcusd",
    "Ethereum": "ethusd",
}


def _stooq_file(symbol: str, stooq_dir: Path = STOOQ_DIR) -> Path:
    # Accept symbol variations like ^spx -> spx_d.csv or btc.v -> btc_v_d.csv
    clean = symbol.lstrip("^").replace(".", "_")
    p1 = stooq_dir / f"{clean}_d.csv"
    if p1.exists():
        return p1
    return stooq_dir / f"{symbol.lstrip('^')}_d.csv"


def _load_stooq_prices(stooq_dir: Path = STOOQ_DIR) -> Optional[pd.DataFrame]:
    """Return wide (Date, SPX, Gold, Bitcoin, Ethereum) closing prices from Stooq CSVs, or None."""
    files = {}
    for asset, sym in STOOQ_SYMBOLS.items():
        f = _stooq_file(sym, stooq_dir)
        if not f.exists() and asset == "Ethereum":
            f = _stooq_file("eth.v", stooq_dir)
        if not f.exists() or f.stat().st_size < 100:
            return None
        files[asset] = f

    wide: Optional[pd.DataFrame] = None
    for asset, f in files.items():
        try:
            df = pd.read_csv(f)
            df = df.rename(columns={"Data": "Date", "Zamkniecie": "Close"})
            if "Date" not in df.columns or "Close" not in df.columns:
                return None
            df = df[["Date", "Close"]].rename(columns={"Close": asset})
            df["Date"] = pd.to_datetime(df["Date"])
            wide = df if wide is None else wide.merge(df, on="Date", how="outer")
        except Exception:
            return None
    return wide


def prepare_real_4assets_dataset(
    raw_csv_path: Path = RAW_4ASSETS_CSV_PATH,
    processed_4assets_path: Path = REAL_4ASSETS_CSV_PATH,
    processed_2assets_path: Path = REAL_2ASSETS_CSV_PATH,
    start_date: str = "2018-06-04",
    end_date: str = "2023-06-04",
    download_if_missing: bool = False,
) -> pd.DataFrame:
    """Prepare the cleaned 4-asset dataset (SPX, Gold, Bitcoin, Ethereum) from Szczepocki (2023).

    Following Section 4 of the paper (p. 50):
    - Sample period: 4 June 2018 to 4 June 2023
    - Only keeps trading days when observations are present for all four time series
      (yielding 1,259 price observations and T = 1,258 daily return observations)
    - Daily logarithmic returns multiplied by 100: y_{i,t} = 100 * log(P_{i,t} / P_{i,t-1})

    Data source priority:
    1. Stooq CSVs in `data/raw/stooq/` (the paper's source; see `STOOQ_SYMBOLS`) — rebuilds
       `raw_csv_path` from them.
    2. Existing `raw_csv_path` (currently built from Yahoo Finance: ^GSPC, GC=F futures,
       BTC-USD, ETH-USD). SPX matches Table 1 exactly; Gold/BTC/ETH differ slightly from the
       paper because of the different gold instrument and crypto daily-close conventions.
    """
    stooq_df = _load_stooq_prices()
    if stooq_df is not None:
        raw_df = stooq_df[
            (stooq_df["Date"] >= pd.to_datetime(start_date))
            & (stooq_df["Date"] <= pd.to_datetime(end_date))
        ]
        raw_df = (
            raw_df[["Date", "SPX", "Gold", "Bitcoin", "Ethereum"]]
            .dropna()
            .sort_values("Date")
            .reset_index(drop=True)
        )
        raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_csv_path, index=False)
    yf_cache_path = DATA_RAW_DIR / "yfinance_4assets_raw.csv"
    if stooq_df is not None:
        pass
    elif not raw_csv_path.exists():
        if yf_cache_path.exists():
            raw_df = pd.read_csv(yf_cache_path, parse_dates=["Date"])
            raw_df = raw_df.rename(
                columns={
                    "^GSPC": "SPX",
                    "GC=F": "Gold",
                    "BTC-USD": "Bitcoin",
                    "ETH-USD": "Ethereum",
                }
            )
            raw_df = raw_df[
                (raw_df["Date"] >= pd.to_datetime(start_date))
                & (raw_df["Date"] <= pd.to_datetime(end_date))
            ]
            raw_df = (
                raw_df[["Date", "SPX", "Gold", "Bitcoin", "Ethereum"]]
                .dropna()
                .sort_values("Date")
                .reset_index(drop=True)
            )
            raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
            raw_df.to_csv(raw_csv_path, index=False)
        elif download_if_missing:
            import yfinance as yf

            tickers = ["^GSPC", "GC=F", "BTC-USD", "ETH-USD"]
            dl = yf.download(tickers, start="2018-06-01", end="2023-06-06", progress=False)["Close"]
            raw_df = dl.reset_index().rename(
                columns={
                    "^GSPC": "SPX",
                    "GC=F": "Gold",
                    "BTC-USD": "Bitcoin",
                    "ETH-USD": "Ethereum",
                }
            )
            raw_df["Date"] = pd.to_datetime(raw_df["Date"])
            raw_df = raw_df[
                (raw_df["Date"] >= pd.to_datetime(start_date))
                & (raw_df["Date"] <= pd.to_datetime(end_date))
            ]
            raw_df = (
                raw_df[["Date", "SPX", "Gold", "Bitcoin", "Ethereum"]]
                .dropna()
                .sort_values("Date")
                .reset_index(drop=True)
            )
            raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
            raw_df.to_csv(raw_csv_path, index=False)
        else:
            raise FileNotFoundError(
                f"Raw 4-asset price file not found at {raw_csv_path}."
            )
    else:
        raw_df = pd.read_csv(raw_csv_path, parse_dates=["Date"])
        raw_df = (
            raw_df[["Date", "SPX", "Gold", "Bitcoin", "Ethereum"]]
            .dropna()
            .sort_values("Date")
            .reset_index(drop=True)
        )

    # Calculate 100 * daily log-returns for each of the 4 assets
    ret_df = pd.DataFrame({"Date": raw_df["Date"].iloc[1:].to_numpy()})
    ret_df["t"] = np.arange(len(ret_df), dtype=int)

    for idx, asset in enumerate(ASSET_NAMES_4, start=1):
        prices = raw_df[asset].to_numpy(dtype=float)
        log_ret_100 = 100.0 * np.log(prices[1:] / prices[:-1])
        ret_df[f"y_{asset}"] = log_ret_100
        ret_df[f"y_{idx}"] = log_ret_100
        ret_df[f"price_{asset}"] = prices[1:]

    processed_4assets_path.parent.mkdir(parents=True, exist_ok=True)
    ret_df.to_csv(processed_4assets_path, index=False)

    # Also save the fast 2-asset real dataset (SPX & Gold)
    ret_2_df = ret_df[
        ["Date", "t", "y_SPX", "y_Gold", "y_1", "y_2", "price_SPX", "price_Gold"]
    ].copy()
    ret_2_df.to_csv(processed_2assets_path, index=False)

    return ret_df


def compute_table1_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Table 1 summary statistics from Szczepocki (2023, p. 51)."""
    rows = []
    for asset in ASSET_NAMES_4:
        col = f"y_{asset}" if f"y_{asset}" in df.columns else asset
        r = df[col].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "Asset": asset,
                "Mean": round(float(np.mean(r)), 3),
                "St. Dev.": round(float(np.std(r, ddof=1)), 3),
                # Population (biased) moment estimators reproduce Table 1 exactly for SPX
                # (skewness -0.77, kurtosis 16.22); non-excess kurtosis as in the paper.
                "Skewness": round(float(stats.skew(r, bias=True)), 3),
                "Kurtosis": round(float(stats.kurtosis(r, fisher=False, bias=True)), 3),
                "Min": round(float(np.min(r)), 3),
                "Q1": round(float(np.percentile(r, 25)), 3),
                "Median": round(float(np.median(r)), 3),
                "Q3": round(float(np.percentile(r, 75)), 3),
                "Max": round(float(np.max(r)), 3),
            }
        )
    return pd.DataFrame(rows)


def load_real_dataset(
    p: int = 4,
    csv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load the cleaned real 4-asset (or 2-asset) log-returns dataset.

    Parameters
    ----------
    p : int
        Number of assets: `4` for full (SPX, Gold, Bitcoin, Ethereum) or `2` for (SPX, Gold).
    csv_path : Path, optional
        Custom path to CSV file.

    Returns
    -------
    dict
        Dictionary containing:
        - `'df'`: pandas DataFrame
        - `'dates'`: pandas Series of timestamps
        - `'y_obs'`: 2D numpy array of shape `(T, p)` ready for `ssm.Bootstrap(ssm=model, data=y_obs)`
        - `'prices'`: 2D numpy array of shape `(T, p)` of asset closing prices
        - `'asset_names'`: tuple of asset names
        - `'table2_params'`: dictionary of Table 2 reference parameters for `p` assets
    """
    if p == 4:
        target_path = csv_path or REAL_4ASSETS_CSV_PATH
        assets = ASSET_NAMES_4
        ref_params = dict(TABLE2_PARAMS_4ASSET)
    elif p == 2:
        target_path = csv_path or REAL_2ASSETS_CSV_PATH
        assets = ASSET_NAMES_2
        ref_params = dict(TABLE2_PARAMS_2ASSET)
    else:
        raise ValueError(f"Supported asset counts for real data are p=4 or p=2, got p={p}.")

    if not target_path.exists():
        prepare_real_4assets_dataset()

    df = pd.read_csv(target_path, parse_dates=["Date"])
    y_cols = [f"y_{a}" for a in assets]
    price_cols = [f"price_{a}" for a in assets]

    return {
        "df": df,
        "dates": pd.to_datetime(df["Date"]),
        "y_obs": df[y_cols].to_numpy(dtype=float),
        "prices": df[price_cols].to_numpy(dtype=float),
        "asset_names": assets,
        "table2_params": ref_params,
        "T": len(df),
        "p": p,
    }


# ---------------------------------------------------------------------------
# 5. Simulated Test Datasets (Fast 2-Asset & Full 4-Asset using Table 2 Params)
# ---------------------------------------------------------------------------
def generate_simulated_chmsv_dataset(
    p: int = 4,
    T: Optional[int] = None,
    params: Optional[Dict[str, float]] = None,
    seed: int = 42,
    init_dist: str = "dirac",
    save_csv: bool = True,
    csv_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Simulate synthetic multivariate returns, hidden states, volatilities, and correlations.

    By default:
    - For `p = 4` (Full 4-Asset Model, 10 states, 30 parameters):
      Uses `TABLE2_PARAMS_4ASSET` and `T = 1258` (matching Section 5 of Szczepocki, 2023).
    - For `p = 2` (Fast 2-Asset Debugging Model, 3 states, 9 parameters):
      Uses `TABLE2_PARAMS_2ASSET` (SPX & Gold) and `T = 500`.
    """
    if T is None:
        T = 1258 if p == 4 else 500

    default_p = TABLE2_PARAMS_4ASSET if p == 4 else TABLE2_PARAMS_2ASSET
    sim_params = dict(default_p)
    if params is not None:
        sim_params.update(params)

    if csv_path is None:
        csv_path = SIM_4ASSETS_CSV_PATH if p == 4 else SIM_2ASSETS_CSV_PATH
    if meta_path is None:
        meta_path = SIM_4ASSETS_META_PATH if p == 4 else SIM_2ASSETS_META_PATH

    np.random.seed(seed)
    model = ChMSV(p=p, init_dist=init_dist, **sim_params)
    x_list, y_list = model.simulate(T)

    # Stack simulated lists into 2D arrays of shape (T, d_x) and (T, p)
    x_true = np.vstack([ np.asarray(xt, dtype=float).reshape(1, -1) for xt in x_list ])
    y_obs = np.vstack([ np.asarray(yt, dtype=float).reshape(1, -1) for yt in y_list ])

    # Transform true states into true return standard deviations (Eq. 6) and correlations (Eq. 7)
    transformed = states_to_volatility_and_correlation(x_true, p=p)
    stdev_true = transformed["stdev"]
    corr_pairs_true = transformed["corr_pairs"]

    data_cols: Dict[str, Any] = {"t": np.arange(T, dtype=int)}
    asset_names = ASSET_NAMES_4[:p]
    for i in range(1, p + 1):
        data_cols[f"y_{i}"] = y_obs[:, i - 1]
        data_cols[f"y_{asset_names[i - 1]}"] = y_obs[:, i - 1]

    state_names = get_state_names(p)
    for k, s_name in enumerate(state_names):
        data_cols[s_name] = x_true[:, k]

    for i in range(1, p + 1):
        data_cols[f"sd_{i}"] = stdev_true[:, i - 1]

    for k, (i, j) in enumerate(get_offdiag_pairs(p)):
        # Store both corr_12 (Table 2 pair notation) and corr_21
        data_cols[f"corr_{j}{i}"] = corr_pairs_true[:, k]

    df = pd.DataFrame(data_cols)

    if save_csv:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        metadata = {
            "p": p,
            "dx": model.dx,
            "num_params": model.num_params,
            "T": T,
            "seed": seed,
            "init_dist": init_dist,
            "asset_names": list(asset_names),
            "state_names": state_names,
            "offdiag_pairs_row_col": [list(pair) for pair in get_offdiag_pairs(p)],
            "offdiag_pairs_table2": [[j, i] for (i, j) in get_offdiag_pairs(p)],
            "true_params": vectors_to_params_dict(
                model.mu_vec, model.phi_vec, model.sigma_vec, p=p
            ),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return df


def load_simulated_chmsv_dataset(
    p: int = 4,
    csv_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load the simulated 4-asset or 2-asset ChMSV test dataset (generating if missing).

    Parameters
    ----------
    p : int
        `4` for the full 4-asset dataset (10 states, 30 params, T=1258) or
        `2` for the fast 2-asset debugging dataset (3 states, 9 params, T=500).

    Returns
    -------
    dict
        Dictionary containing:
        - `'df'`: full pandas DataFrame
        - `'y_obs'`: 2D numpy array of shape `(T, p)` of simulated log-returns
        - `'x_true'`: 2D numpy array of shape `(T, d_x)` of true hidden states (h, q)
        - `'h_true'`: 2D numpy array of shape `(T, p)` of true log-volatilities
        - `'q_true'`: 2D numpy array of shape `(T, p*(p-1)//2)` of true Cholesky states
        - `'stdev_true'`: 2D numpy array of shape `(T, p)` of true return stdev (Eq. 6)
        - `'corr_true'`: 2D numpy array of shape `(T, p*(p-1)//2)` of true dynamic correlations (Eq. 7)
        - `'true_params'`: dictionary of true parameters (Table 2 values)
        - `'state_names'`: list of state variable names
    """
    target_csv = csv_path or (SIM_4ASSETS_CSV_PATH if p == 4 else SIM_2ASSETS_CSV_PATH)
    target_meta = meta_path or (
        SIM_4ASSETS_META_PATH if p == 4 else SIM_2ASSETS_META_PATH
    )

    if not target_csv.exists():
        generate_simulated_chmsv_dataset(
            p=p, save_csv=True, csv_path=target_csv, meta_path=target_meta
        )

    df = pd.read_csv(target_csv)
    state_names = get_state_names(p)
    y_cols = [f"y_{i}" for i in range(1, p + 1)]
    sd_cols = [f"sd_{i}" for i in range(1, p + 1)]
    corr_cols = [f"corr_{j}{i}" for (i, j) in get_offdiag_pairs(p)]

    default_params = dict(TABLE2_PARAMS_4ASSET if p == 4 else TABLE2_PARAMS_2ASSET)
    if target_meta.exists():
        with open(target_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
            true_params = meta.get("true_params", default_params)
    else:
        true_params = default_params

    x_true = df[state_names].to_numpy(dtype=float)

    return {
        "df": df,
        "y_obs": df[y_cols].to_numpy(dtype=float),
        "x_true": x_true,
        "h_true": x_true[:, :p],
        "q_true": x_true[:, p:],
        "stdev_true": df[sd_cols].to_numpy(dtype=float),
        "corr_true": df[corr_cols].to_numpy(dtype=float),
        "true_params": true_params,
        "state_names": state_names,
        "asset_names": ASSET_NAMES_4[:p],
        "T": len(df),
        "p": p,
        "dx": p * (p + 1) // 2,
    }


# ---------------------------------------------------------------------------
# 6. Role 1 Data Visualization Helpers (Figure 1 Reproduction & Sandbox Preview)
# ---------------------------------------------------------------------------
def plot_figure1_real_data(
    real_data: Optional[Dict[str, Any]] = None,
    save_path: Optional[Path] = PLOTS_DIR / "fig1_real_4assets_prices_and_returns.png",
) -> plt.Figure:
    """Reproduce Figure 1 of Szczepocki (2023, p. 51): Prices (left) and Log-Returns*100 (right)."""
    if real_data is None:
        real_data = load_real_dataset(p=4)

    dates = real_data["dates"]
    prices = real_data["prices"]
    y_obs = real_data["y_obs"]
    assets = real_data["asset_names"]

    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex="col")
    colors = ["#1F4E79", "#B8860B", "#D95F02", "#5E3C99"]

    for i, (asset, color) in enumerate(zip(assets, colors)):
        ax_p = axes[i, 0]
        ax_r = axes[i, 1]

        ax_p.plot(dates, prices[:, i], color=color, lw=1.1)
        ax_p.set_ylabel(asset, fontsize=10, fontweight="bold")
        ax_p.grid(True, alpha=0.25)

        ax_r.plot(dates, y_obs[:, i], color="#111111", lw=0.65)
        ax_r.set_ylabel(f"{asset} $y_{{{i+1},t}}$", fontsize=9)
        ax_r.grid(True, alpha=0.25)

    axes[0, 0].set_title("Asset Prices (USD / Index)", fontsize=11, fontweight="bold")
    axes[0, 1].set_title(
        "Daily Logarithmic Returns $\\times 100$ ($T = 1258$)",
        fontsize=11,
        fontweight="bold",
    )
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(
        "Reproduction of Figure 1 (Szczepocki, 2023): SPX, Gold, Bitcoin & Ethereum (4 June 2018 – 4 June 2023)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def plot_simulated_datasets_overview(
    sim4: Optional[Dict[str, Any]] = None,
    sim2: Optional[Dict[str, Any]] = None,
    save_path: Optional[Path] = PLOTS_DIR / "simulated_chmsv_datasets_overview.png",
) -> plt.Figure:
    """Plot ground-truth volatilities and dynamic correlations of the simulated ChMSV datasets."""
    if sim4 is None:
        sim4 = load_simulated_chmsv_dataset(p=4)
    if sim2 is None:
        sim2 = load_simulated_chmsv_dataset(p=2)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))

    # Top-left: 4-Asset True Standard Deviations (Eq. 6)
    ax00 = axes[0, 0]
    t4 = np.arange(sim4["T"])
    for i, name in enumerate(sim4["asset_names"]):
        ax00.plot(t4, sim4["stdev_true"][:, i], lw=1.0, label=f"{name} $\\sqrt{{\\Sigma_{{t,{i+1}{i+1}}}}}$")
    ax00.set_title("4-Asset Simulated Ground-Truth Volatilities (Eq. 6, $T=1258$)", fontsize=10, fontweight="bold")
    ax00.set_ylabel("Stdev $(y_{i,t} \\mid \\mathcal{F}_{t-1})$")
    ax00.legend(loc="upper right", fontsize=8)
    ax00.grid(True, alpha=0.25)

    # Top-right: 4-Asset True Dynamic Correlations with SPX (Eq. 7)
    ax01 = axes[0, 1]
    pair_labels_4 = [
        "SPX–Gold $\\rho_{12,t}$",
        "SPX–BTC $\\rho_{13,t}$",
        "SPX–ETH $\\rho_{14,t}$",
        "Gold–BTC $\\rho_{23,t}$",
        "Gold–ETH $\\rho_{24,t}$",
        "BTC–ETH $\\rho_{34,t}$",
    ]
    for k, label in enumerate(pair_labels_4):
        ax01.plot(t4, sim4["corr_true"][:, k], lw=0.9, label=label)
    ax01.set_title("4-Asset Simulated Ground-Truth Dynamic Correlations (Eq. 7)", fontsize=10, fontweight="bold")
    ax01.set_ylabel("Correlation $\\rho_{ij,t}$")
    ax01.legend(loc="lower right", fontsize=7.5, ncol=2)
    ax01.grid(True, alpha=0.25)

    # Bottom-left: Fast 2-Asset True Volatilities (SPX & Gold, T=500)
    ax10 = axes[1, 0]
    t2 = np.arange(sim2["T"])
    ax10.plot(t2, sim2["stdev_true"][:, 0], color="#1F4E79", lw=1.2, label="SPX $\\sqrt{\\Sigma_{t,11}}$")
    ax10.plot(t2, sim2["stdev_true"][:, 1], color="#B8860B", lw=1.2, label="Gold $\\sqrt{\\Sigma_{t,22}}$")
    ax10.set_title("Fast 2-Asset Simulated Ground-Truth Volatilities ($p=2, T=500$)", fontsize=10, fontweight="bold")
    ax10.set_xlabel("Time Step $t$")
    ax10.set_ylabel("Stdev $(y_{i,t} \\mid \\mathcal{F}_{t-1})$")
    ax10.legend(loc="upper right", fontsize=8.5)
    ax10.grid(True, alpha=0.25)

    # Bottom-right: Fast 2-Asset True Dynamic Correlation (SPX–Gold, T=500)
    ax11 = axes[1, 1]
    ax11.plot(t2, sim2["corr_true"][:, 0], color="#C44E52", lw=1.2, label="True SPX–Gold $\\rho_{12,t}$")
    ax11.set_title("Fast 2-Asset Simulated Ground-Truth Dynamic Correlation ($p=2, T=500$)", fontsize=10, fontweight="bold")
    ax11.set_xlabel("Time Step $t$")
    ax11.set_ylabel("Correlation $\\rho_{12,t}$")
    ax11.legend(loc="upper right", fontsize=8.5)
    ax11.grid(True, alpha=0.25)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# CLI Entry Point for Role 1 (Data & Base ChMSV Model Setup)
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute Role 1 (Data & Base ChMSV Model Setup): datasets, transformations, and overview plots."""
    print("=" * 78)
    print("SMC ChMSV Project — Role 1: Data & Base ChMSV Model Setup (Szczepocki, 2023)")
    print("=" * 78)

    # 1. Prepare and verify real 4-asset dataset (4 June 2018 – 4 June 2023, T = 1258)
    real_df = prepare_real_4assets_dataset()
    real_data = load_real_dataset(p=4)
    print(
        f"\n[M1] Cleaned Real 4-Asset Dataset saved to: {REAL_4ASSETS_CSV_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"     Observations: T = {real_data['T']} trading days "
        f"({real_df['Date'].iloc[0]} to {real_df['Date'].iloc[-1]})"
    )
    print("\n     Table 1 Summary Statistics Verification (100 * daily log-returns):")
    stats_df = compute_table1_summary_stats(real_df)
    print(stats_df.to_string(index=False))

    # 2. Generate and verify simulated test datasets (2-asset fast & 4-asset full)
    sim2_df = generate_simulated_chmsv_dataset(p=2, T=500, seed=42, save_csv=True)
    sim4_df = generate_simulated_chmsv_dataset(p=4, T=1258, seed=42, save_csv=True)
    sim2 = load_simulated_chmsv_dataset(p=2)
    sim4 = load_simulated_chmsv_dataset(p=4)

    print(
        f"\n[M1] Fast 2-Asset Simulated Dataset saved to: {SIM_2ASSETS_CSV_PATH.relative_to(PROJECT_ROOT)}"
        f" (T = {sim2['T']}, {sim2['dx']} states, {3 * sim2['dx']} parameters)"
    )
    print(
        f"[M1] Full 4-Asset Simulated Dataset saved to: {SIM_4ASSETS_CSV_PATH.relative_to(PROJECT_ROOT)}"
        f" (T = {sim4['T']}, {sim4['dx']} states, {3 * sim4['dx']} parameters)"
    )

    # 3. Verify `ChMSV` compatibility with `particles.state_space_models.Bootstrap`
    model_4 = ChMSV4Asset()
    fk_4 = ssm.Bootstrap(ssm=model_4, data=real_data["y_obs"])
    x0_sample = fk_4.M0(16)
    x1_sample = fk_4.M(1, x0_sample)
    logG_sample = fk_4.logG(0, x0_sample, x1_sample)
    trans_sample = model_4.transform_states(x1_sample)
    print(
        f"\n[M1/M2] Verified `ChMSV4Asset` in `particles`:"
        f" M0 shape={x0_sample.shape}, M1 shape={x1_sample.shape},"
        f" logG shape={logG_sample.shape}, stdev shape={trans_sample['stdev'].shape},"
        f" corr_pairs shape={trans_sample['corr_pairs'].shape}"
    )

    # 4. Save Figure 1 reproduction & simulated overview plots
    fig1 = plot_figure1_real_data(real_data)
    plt.close(fig1)
    fig_sim = plot_simulated_datasets_overview(sim4, sim2)
    plt.close(fig_sim)
    print(
        "[M1/M2] Saved overview plots to `plots/fig1_real_4assets_prices_and_returns.png` "
        "and `plots/simulated_chmsv_datasets_overview.png`"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()

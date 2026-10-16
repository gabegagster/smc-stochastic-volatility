# SMC Stochastic Volatility Project

This repository contains the codebase for our Dec 17 project on Stochastic Volatility using the `particles` package in Python.

## Project Structure
```text
├── data/
│   ├── raw/               <- The original, immutable S&P 500 data.
│   └── processed/         <- Simulated sandbox data and cleaned datasets.
├── notebooks/             <- Jupyter notebooks for exploration and our final report.
├── plots/                 <- Generated graphics and figures for the presentation.
├── src/                   <- Source code for use in this project.
│   ├── __init__.py
│   ├── base_model.py      <- M1: Setup & Sandbox Data (Gabriel)
│   ├── filtering.py       <- M2: Bootstrap, Guided, and APF (Volunteer 1)
│   ├── smoothing.py       <- M3: FFBS and two-filter smoothing (Volunteer 2)
│   └── estimation.py      <- M4: PMCMC and SMC^2 (Volunteer 3)
├── .gitignore             <- Specifies intentionally untracked files to ignore.
├── requirements.txt       <- The dependencies file for reproducing the environment.
└── README.md              <- The top-level README for developers using this project.
```

## Getting Started
1. Clone this repository.
2. Create a virtual environment: `python -m venv .venv`
3. Activate the environment (Mac/Linux: `source .venv/bin/activate`, Windows: `.venv\Scripts\activate`)
4. Install dependencies: `pip install -r requirements.txt`

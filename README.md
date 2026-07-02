# conformal_prediction

Limited repository for a master's thesis on conformal prediction for time-series forecasting in the energy sector

This snapshot contains reusable utilities, model tuning scripts, HPC-oriented experiment runners, and a small notebook example. The code supports multiple regression and probabilistic forecasting workflows, including classical conformal prediction, quantile-based methods, Bayesian approaches, and TCN-based pipelines.

## Overview

- Thesis topic: Conformal prediction in the energy sector for efficient uncertainty estimation
- Main methods explored: Conformal predictive systems, QCP, BART, boosting, k-NN, linear baselines, and TCN variants

## Repository Layout

- `src/`: core data loading, metrics, models, conformal prediction, and tuning utilities
- `hpc/`: larger experiment scripts intended for batch or cluster execution on a hpc
- `toy_example.ipynb`: compact notebook example for quick experimentation
- `requirements.txt`: Python dependencies for the code in this snapshot

## Setup

1. Create and activate a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Verify that the expected datasets are available locally.

### Environment Notes

- Python version: 3.11.14
- Package manager or environment tool: conda


## Data

- Raw data is confidential
- An external API call can be made to Energi Data Service for publicly available data

## Usage

### Notebook

Open `toy_example.ipynb` to run a lightweight example and inspect the core workflow.

### Scripts

- Core utilities: import from `src/`
- Experiment scripts: run files in `hpc/` for tuning or larger-scale experiments

Example entry points:


## Development Notes

- Several scripts are written for long-running experiments and may assume access to local datasets or HPC storage.
- Some dependencies are only needed for specific model families, so a minimal environment may still work for a subset of the repository.
- The actual data processing, model parameters and analysis is conducted in notebooks not available here.

## Contact

- Author: Jacob Skaarup and Martha Kofod
- Institution: DTU, Denmark

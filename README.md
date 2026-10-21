# TARGet

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![Paper Status](https://img.shields.io/badge/Paper-Under%20Review-orange)
![Release Type](https://img.shields.io/badge/Release-Code%20Only-lightgrey)

TARGet is an open-source topology-aware RF circuit modeling framework for RF circuits, including low-noise amplifiers and phase shifters. It combines sub-circuit connectivity-aware models with Graph Neural Networks to support unified learning across multiple circuit topologies.

## Summary

TARGet is designed for RF functional modeling under limited training data. The method uses:

- S-parameter-based sub-circuit representations
- explicit topology encoding
- a fusion-based architecture that combines CCI-style subnetworks and a GNN branch

The associated paper is currently under review.

## Repository Scope

This repository contains code only. No dataset, processed data, or derived predictions are included.
The dataset is not published because its distribution is restricted under Non-Disclosure Agreement (NDA) protocols.

The expected external dataset layout is:

```text
dataset/
  <architecture>/
    main/
      <topology>/
        train_*.csv
        test_*.csv
```

Expected CSV columns:

- LNA:

```text
freq w in_s11_abs in_s11_ph in_s12_abs in_s12_ph in_s21_abs in_s21_ph in_s22_abs in_s22_ph out_s11_abs out_s11_ph out_s12_abs out_s12_ph out_s21_abs out_s21_ph out_s22_abs out_s22_ph GA GP GT NF b1f kf topology_index
```

- Phase shifter:

```text
freq s2 s1_s11_abs s1_s11_ph s1_s12_abs s1_s12_ph s1_s21_abs s1_s21_ph s1_s22_abs s1_s22_ph s2_s11_abs s2_s11_ph s2_s12_abs s2_s12_ph s2_s21_abs s2_s21_ph s2_s22_abs s2_s22_ph S11m S21m S21p S22m topology_index
```

## Environment

The canonical environment for this release is [`environment.yml`](environment.yml).

Main dependencies:

- Python 3.10
- TensorFlow / Keras
- NumPy / pandas / scikit-learn
- Matplotlib / seaborn
- Spektral
- PyTorch Geometric

## Quick Start

```bash
conda env create -f environment.yml
conda activate target
python code/main.py
```

Before running, place the external dataset under the expected `dataset/` directory and adjust [`code/config.py`](code/config.py) if needed.

## Reproducing This Release

The default configuration in [`code/config.py`](code/config.py) uses:

- `main_arch="LNA"`
- `model_types=["FCN", "CCI", "GNN", "CCIGNN-Sep"]`
- `selected_topologies=[0, 1] # Both CS and Cascode`
- `targets=["GA"] # GP, NF`
- `freq_mode="uniform"`
- `freq_count=40`

## Outputs

Running `code/main.py` creates experiment outputs under:

- `results/` for logs, CSV summaries, and analysis tables
- `weights/` for saved models
- `images/` for generated figures

## Repository Layout

- `code/main.py`: main experiment entry point
- `code/config.py`: experiment configuration
- `code/data/`: loading, preprocessing, diagnostics, adapters
- `code/models/`: FCN, CCI-NN, GNN, and TARGet models
- `code/training/`: training loop and evaluation helpers
- `code/analysis/`: plots and summary tables
- `code/utils/`: I/O, plotting, seeding, and shared utilities

## Reproducibility Notes

- Random seeds are controlled through `base_seed` in `code/config.py`.
- This release is intended for external datasets that match the expected schema.
- GPU-backed execution is recommended for larger runs.

## License

Released under the MIT License. See [LICENSE](LICENSE).

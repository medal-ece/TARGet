# TARGet

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2609.22165-b31b1b)](https://arxiv.org/abs/2609.22165)
![Release Type](https://img.shields.io/badge/Release-Code%20Only-lightgrey)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)

TARGet is an open-source topology-aware RF circuit modeling framework for RF circuits, including low-noise amplifiers and phase shifters. It combines sub-circuit connectivity-aware models with Graph Neural Networks to support unified learning across multiple circuit topologies.

## Summary

TARGet is designed for RF functional modeling under limited training data. The method uses:

- S-parameter-based sub-circuit representations
- explicit topology encoding
- a fusion-based architecture that combines CCI-style subnetworks and a GNN branch

**Published Paper:** [TARGet: Topology-Aware Fusion-based Radio Frequency Circuit Functional Modeling using Graph Neural Networks](https://arxiv.org/abs/2609.22165). Citation information is in [CITATION.cff](CITATION.cff).

## Installation

Use Python 3.10 and the supplied Conda environment:

```bash
conda env create -f environment.yml
conda activate target
```

The environment includes TensorFlow/Keras, Spektral, and PyTorch/PyG. For GPU execution with the pinned TensorFlow version, use a compatible Linux or WSL2 setup; native Windows GPU support requires TensorFlow 2.10 or earlier. See the [TensorFlow installation guide](https://www.tensorflow.org/install/pip).

## Dataset

This repository contains code only. No dataset, processed data, or derived predictions are included. The dataset is not published because its distribution is restricted under Non-Disclosure Agreement (NDA) protocols.

The expected external dataset layout is:

```text
dataset/v1/<architecture>/main/<topology>/
  train_*.csv
  test_*.csv
```

Each CSV needs a header, frequency in Hz, circuit features, targets, and `topology_index`. S-parameter phases are in degrees. Use the following schemas:

**LNA**

```text
freq w in_s11_abs in_s11_ph in_s12_abs in_s12_ph in_s21_abs in_s21_ph in_s22_abs in_s22_ph out_s11_abs out_s11_ph out_s12_abs out_s12_ph out_s21_abs out_s21_ph out_s22_abs out_s22_ph GA GP GT NF b1f kf topology_index
```

**Phase shifter**

```text
freq s2 s1_s11_abs s1_s11_ph s1_s12_abs s1_s12_ph s1_s21_abs s1_s21_ph s1_s22_abs s1_s22_ph s2_s11_abs s2_s11_ph s2_s12_abs s2_s12_ph s2_s21_abs s2_s21_ph s2_s22_abs s2_s22_ph S11m S21m S21p S22m topology_index
```

Selected frequencies must occur in both the CSVs and [code/freq_list.py](code/freq_list.py).

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

## Run

Run commands from the repository root. Choose the architecture, targets, topologies, and training settings in [code/config.py](code/config.py), then:

```bash
python code/main.py
```

The default runs LNA gain (`GA`) across both topologies, using all four model families, 40 frequencies, and a training-data utilization sweep. For phase shifters, set `main_arch="phase_shifter"`, `targets=["S11m"]`, and `selected_topologies=[0, 1, 2, 3]` together.

For CPU execution without the entry-point GPU check:

```bash
python -c "import sys; sys.path.insert(0, 'code'); from main import main; main()"
```

On headless systems, set `MPLBACKEND=Agg` and `plot_show=False`. Caching can be disabled with `use_dataset_cache=False`.

### Zero-shot evaluation

```bash
python code/run_experiment.py
```

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

## License

Released under the MIT License. See [LICENSE](LICENSE).

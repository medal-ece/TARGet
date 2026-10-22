# TARGet

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2609.22165-b31b1b)](https://arxiv.org/abs/2609.22165)

TARGet is a topology-aware RF circuit modeling framework for low-noise amplifiers (LNAs) and phase shifters. It combines sub-circuit neural networks with a graph neural network and includes FCNN, CCI-NN, and GNN baselines.

**Paper:** [TARGet: Topology-Aware Fusion-based Radio Frequency Circuit Functional Modeling using Graph Neural Networks](https://arxiv.org/abs/2609.22165). Citation information is in [CITATION.cff](CITATION.cff).

## Installation

Use Python 3.10 and the supplied Conda environment:

```bash
conda env create -f environment.yml
conda activate target
```

The environment includes TensorFlow/Keras, Spektral, and PyTorch/PyG. For GPU execution with the pinned TensorFlow version, use a compatible Linux or WSL2 setup; native Windows GPU support requires TensorFlow 2.10 or earlier. See the [TensorFlow installation guide](https://www.tensorflow.org/install/pip).

## Dataset

Datasets are not included because their distribution is restricted by NDA. Place prepared CSV files under:

```text
dataset/v1/<architecture>/main/<topology>/
  train_*.csv
  test_*.csv
```

| Architecture | Combined topology folder | Targets |
| --- | --- | --- |
| `LNA` | `combo_0_1` | `GA`, `GP`, `NF` |
| `phase_shifter` | `combo_0_1_2_3` | `S11m` |

Each CSV needs a header, frequency in Hz, circuit features, targets, and `topology_index`. S-parameter phases are in degrees. Use the following schemas:

**LNA**

```text
freq w in_s11_abs in_s11_ph in_s12_abs in_s12_ph in_s21_abs in_s21_ph in_s22_abs in_s22_ph out_s11_abs out_s11_ph out_s12_abs out_s12_ph out_s21_abs out_s21_ph out_s22_abs out_s22_ph GA GP GT NF b1f kf topology_index
```

**Phase shifter**

```text
freq s2 s1_s11_abs s1_s11_ph s1_s12_abs s1_s12_ph s1_s21_abs s1_s21_ph s1_s22_abs s1_s22_ph s2_s11_abs s2_s11_ph s2_s12_abs s2_s12_ph s2_s21_abs s2_s21_ph s2_s22_abs s2_s22_ph S11m S21m S21p S22m topology_index
```

For LNA, topology IDs are `0` (CS) and `1` (Cascode). For phase shifters, IDs `0`-`3` correspond to switch states `(0,0)`, `(0,1)`, `(1,0)`, and `(1,1)`. Inputs must match these supported circuit structures. Selected frequencies must occur in both the CSVs and [code/freq_list.py](code/freq_list.py).

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

The default preset, `lna_heldout_primary_10freq`, evaluates GA, GP, and NF at 10 frequencies and full utilization. Controls for targets and models are at the top of [code/run_experiment.py](code/run_experiment.py).

Place its prepared data under:

```text
dataset/v2_inf_cap_ind/LNA/
  main/combo_0_1/       # train_*.csv and test_*.csv
  heldout/combo_0_1/    # test_*.csv
```

The primary split holds out `inf_cap_ind` wherever it occurs on either matching side: input IDs 61-63 or output IDs 57-59. The conversion utilities in [code/ds_conv.py](code/ds_conv.py) use the original archive/structured-data conventions; they are unnecessary when the prepared splits are already available. Regeneration can replace existing generated files.

## Outputs

- `results/main/`: metrics, predictions, analysis, and diagnostic plots.
- `weights/main/`: saved models and metadata.
- Dataset `.cache/` folders: cached inputs when enabled.

MAE/RMSE logs use standardized target units; saved predictions use original units. NMAE divides MAE by the evaluated target range. Timing records distinguish model fitting (`training_time_sec`) from the full experiment loop (`loop_time_sec`).

Repeated runs can append records or reuse weight filenames. Preserve previous outputs when needed. Keep private datasets out of commits: the current `.gitignore` does not exclude `dataset/` or `Files/`.

## License

[MIT](LICENSE).

# TARGet

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
[![Paper](https://img.shields.io/badge/arXiv-2609.22165-b31b1b)](https://arxiv.org/abs/2609.22165)
![Release Type](https://img.shields.io/badge/Release-Code%20Only-lightgrey)

TARGet is an open-source topology-aware RF circuit modeling framework for RF circuits, including low-noise amplifiers and phase shifters. It combines sub-circuit connectivity-aware models with Graph Neural Networks to support unified learning across multiple circuit topologies.

## Summary

TARGet is designed for RF functional modeling under limited training data. The method uses:

- S-parameter-based sub-circuit representations
- explicit topology encoding
- a fusion-based architecture that combines CCI-style subnetworks and a GNN branch

Paper: [TARGet: Topology-Aware Fusion-based Radio Frequency Circuit Functional Modeling using Graph Neural Networks](https://arxiv.org/abs/2609.22165), by Soroosh Noorzad, Sebastian Bodero, and Morteza Fayazi. See [CITATION.cff](CITATION.cff) for citation metadata.

This is a code-only research release. It preserves the transferred FuNToM implementation; it does not silently correct historical modeling behavior. Running on a compatible user-supplied dataset is distinct from reproducing the paper's numerical results.

## Repository Scope

This repository contains code only. No dataset, processed data, or derived predictions are included.
The dataset is not published because its distribution is restricted under Non-Disclosure Agreement (NDA) protocols.

The current default is `dataset_version="v1"`. Run commands from the repository root. Paths are resolved relative to this checkout, so the training workflow does not require the author's machine, FuNToM checkout, simulator, or private `Files/` archive once prepared CSV inputs are supplied.

The default external dataset layout is:

```text
dataset/v1/
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

## Installation and platform support

The declared dependency specification is [environment.yml](environment.yml), using Python 3.10, TensorFlow 2.14.1, Keras 2.14.0, and Spektral 1.3.1.

```bash
conda env create -f environment.yml
conda activate target
python -m pip check
python -B -c "import sys; sys.path.insert(0, 'code'); import main, ds_conv; print('Imports OK')"
```

`environment.yml` explicitly includes `torch==2.4.1` alongside `torch-geometric==2.6.1` because `code/models/gnn.py` imports PyG, which requires PyTorch. See the [PyG 2.6.1 installation documentation](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html). The networks themselves use TensorFlow/Keras; PyG's optional compiled extensions are not needed by the current import path. This PyTorch pin is an installation dependency choice, not a claim about the version used for historical paper runs. A fresh environment with this pin has not been validated.

Model-diagram rendering may require `pydot` and the Graphviz executable. The training code catches diagram-rendering errors and continues, so these are optional for training. Other diagnostic and analysis plots still require Matplotlib and a writable output directory.

For GPU use, install the drivers and CUDA/cuDNN components compatible with the pinned TensorFlow release. TensorFlow versions after 2.10 do not support CUDA GPU execution on native Windows; use a compatible Linux/WSL2 environment for GPU execution with the declared TensorFlow version. See the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip), including its links to older-version requirements. Its latest-version CUDA instructions should not be applied blindly to this pinned environment.

On headless machines set `MPLBACKEND=Agg` in the shell and use `plot_show=False`. Full-frequency graph runs can require substantial RAM because graph tensors are built in memory. Dataset caching is enabled by default and also uses disk/RAM; disable it through `use_dataset_cache=False` when appropriate.

The current transferred source has passed import checks in an existing environment using TensorFlow/Keras 2.10.0 and Spektral 1.3.1. A fresh installation of the exact declared environment has not been validated in this review. This is not a cross-platform certification or a record of every historical experiment environment.

## Running with your own datasets

### Supported inputs and circuit families

This is not a general circuit/netlist importer. Your data must match the existing LNA or phase-shifter feature schema above and the graph definitions in `code/models/gnn.py`.

| Setting | LNA | Phase shifter |
| --- | --- | --- |
| `main_arch` | `"LNA"` | `"phase_shifter"` |
| `topology_index` | 0 = CS; 1 = Cascode | 0-3 correspond to `sw1_0_sw2_0`, `sw1_0_sw2_1`, `sw1_1_sw2_0`, `sw1_1_sw2_1` |
| Example `selected_topologies` | `[0, 1]` | `[0, 1, 2, 3]` |
| Combined directory | `combo_0_1` | `combo_0_1_2_3` |
| Single-topology directories | `CS`, `Cascode` | Corresponding `sw1_*_sw2_*` name |
| Targets used by the paper | `GA`, `GP`, `NF` | `S11m` |

For example, the default LNA files go in:

```text
dataset/v1/LNA/main/combo_0_1/
  train_01.csv
  test_01.csv
```

Use CSV headers, numeric columns, frequency in Hz, and S-parameter phases in degrees. Preserve consistent physical units across train/test. With `cartesian=True`, the implementation constructs the required real/imaginary S-parameter features from the magnitude/phase columns. Full schema inputs above are the safest contract; phase-shifter diagnostics also read `S21m` and `S21p`, even when training only `S11m`. Supply finite, non-missing required features and targets, with enough varying samples at every selected frequency for validation and evaluation. This release does not provide a general imputation pipeline for arbitrary missing inputs.

Frequency selection uses the fixed list in `code/freq_list.py`. `uniform` means uniform selection from that list, not from frequencies discovered in your CSVs. Manual frequencies must also occur in that list and in both input splits. Arbitrary new frequency grids and new circuit graph structures require future code adaptation; they are not currently configuration-only capabilities.

Keep your train/test split independent. For a topology-held-out experiment, establish the split before using these inputs and verify that the intended topology does not appear in the training or ordinary test pool. Matching IDs only identify physical structures if their mapping is valid for your dataset.

### Default execution

```bash
python code/main.py
```

Defaults select LNA, GA, both high-level LNA topologies, the four paper model families, 40 uniform frequencies, and the utilization sweep `[0.01, 0.03, 0.05, 0.2, 0.5, 1.0]`.

The direct entry point calls a GPU diagnostic that exits when TensorFlow was built with CUDA but no GPU is visible. CPU execution through the existing `main(config=...)` API avoids that entry-point diagnostic. This does not require changing repository code:

```bash
python -c "import sys; sys.path.insert(0, 'code'); from config import Config; from main import main; c=Config(); c.plot_show=False; c.use_dataset_cache=False; main(config=c)"
```

This uses the same default dataset and full experiment configuration; it is not a small smoke test. For a smaller user-selected run, configure `model_types`, `utilization`, `freq_count`, and `ep_count` through this API before invoking `main`. Such a run does not reproduce the full paper protocol.

Other dataset layouts are already supported:

- `dataset_version="legacy"`: `dataset/<architecture>/main/<topology>/`.
- Another version string: `dataset/<version>/<architecture>/<split>/<topology>/`.
- A nonempty `dataset_variant` adds a directory between version and architecture.

Set `main_arch`, `targets`, and `selected_topologies` explicitly together when selecting phase-shifter data. Changing only the architecture on a `Config()` instance does not recalculate the class-defined LNA target/topology defaults. For example:

```bash
python -c "import sys; sys.path.insert(0, 'code'); from config import Config; from main import main; c=Config(); c.main_arch='phase_shifter'; c.targets=['S11m']; c.selected_topologies=[0,1,2,3]; c.plot_show=False; main(config=c)"
```

## Paper experiment mapping

The [arXiv paper, Section 4](https://arxiv.org/html/2609.22165v1), specifies 40 frequencies for the original LNA evaluation and 10 uniformly selected frequencies at full utilization for the primary zero-shot experiment. The model names in this code are `FCN` (FCNN), `CCI` (CCI-NN), `GNN`, and `CCIGNN-Sep` (TARGet).

| Paper evidence | Existing execution path |
| --- | --- |
| LNA utilization/frequency results and ordinary-test tables | `main.py`, GA/GP/NF separately, default four models, 40 frequencies and utilization sweep |
| Normalized tolerance evaluation | `analysis/plots.py`, using saved prediction samples |
| Phase-shifter input return loss | `main.py` with architecture, target and topology settings shown above |
| Primary matching-network zero-shot evaluation | `run_experiment.py`, default preset `lna_heldout_primary_10freq` |
| Timing records | `training/experiment.py` and `analysis/reports.py` |

The current phase-shifter frequency count must be matched to the intended original run; the paper does not specify its count explicitly. Historical exact commit/configuration/environment records are incomplete. External MATLAB post-processing described in the audit is not included here, so the included Python plots/tables are not a claim of identical final paper figure styling or recovery of the exact data-efficiency interpolation procedure.

### Primary zero-shot evaluation

```bash
python code/run_experiment.py
```

The source launcher has editable controls `PRESET_NAME`, `TARGETS`, `MODEL_TYPES`, and `EXPERIMENT_ID`. It does not implement `--preset`, `--list`, or other command-line flags. The existing API can select a target/model subset without editing it:

```bash
python -c "import sys; sys.path.insert(0, 'code'); from experiments.presets import run_preset; run_preset('lna_heldout_primary_10freq', targets=['GA'], model_types=['FCN','CCI','GNN','CCIGNN-Sep'])"
```

The primary preset selects GA/GP/NF, the four paper models, 10 uniform frequencies, 100% utilization, one repetition, and dataset `v2_inf_cap_ind`:

```text
dataset/v2_inf_cap_ind/LNA/
  main/combo_0_1/
    train_*.csv
    test_*.csv
  heldout/combo_0_1/
    test_*.csv
```

The primary physical structure is named `inf_cap_ind` in the registry: input matching IDs 61-63 or output matching IDs 57-59, using the either-side rule. These IDs differ from the older raw-file-ID held-out split. Structural metadata columns are `in_topology_id` and `out_topology_id`; they support conversion/provenance rather than additional numeric model inputs. For your own data, do not reuse those numeric IDs without confirming that they describe the intended structures.

The supplied conversion script can derive the primary combined split from a compatible local structured v5 dataset:

```bash
python code/ds_conv.py --mode lna-heldout-v2-inf-cap-ind-combo
```

It expects source chunks under `dataset/v5/LNA/main/combo_0_1/`, containing the structural ID columns. It is not an arbitrary CSV importer. Its write path replaces existing generated chunks, metadata, and caches in the destination; use it only for deliberate dataset regeneration. Other converter modes depend on the original `Files/` archive layout and source numbering/row-order assumptions. That private archive is neither required for training on prepared CSVs nor distributed here.

Auxiliary held-out and cross-family converter modes were transferred from FuNToM as extensions; they are not interchangeable with the primary paper experiment. The archived audit mentions a `v2_corrected_inf_cap_ind` name; its equivalence to a newly generated dataset is not guaranteed merely by using the current `v2_inf_cap_ind` preset.

## Outputs

All paths are rooted at this checkout and need write permission:

- `results/main/<architecture>/<run>/`: logs, ordinary/held-out metrics, predictions, dataset reports, and analysis.
- `weights/main/<dataset-context>/<architecture>/<topology>/`: saved models and `.meta.json` sidecars.
- Per-run `diagnostics/`: most input/preprocessing figures; some utilities also use root `images/`.
- `.cache/` beneath an input dataset directory when caching is enabled.

Primary outputs include `all_repetitions.csv`, `all_predictions.csv`, `heldout_results.csv`, `all_heldout_predictions.csv`, and `dataset_report.json`. Held-out comparisons are under `analysis/heldout_comparison/`. Additional runtime/input summaries are under `analysis/rebuttal/`, preserving the source implementation's directory name. `run_results.csv` includes `training_time_sec` where measured; the existing runtime summaries aggregate `loop_time_sec` instead.

An existing `EXPERIMENT_ID` supports deliberate partial reruns. Completed configurations are not automatically deduplicated, so rerunning them can append duplicate records. Existing saved-model names can also collide across repeated fits; use a separate checkout/output archive when preserving independent runs.

## Implementation conventions and reproducibility limits

These describe current code, not corrections to the published article:

- Input/output scaling and clipping statistics use the main training DataFrame before validation splitting and utilization subsampling. Low-utilization fits therefore still use full-pool preprocessing statistics, including output-scaling statistics. Held-out evaluation reuses main-training statistics.
- MAE/RMSE logs are in standardized output units; prediction samples are inverse-transformed. NMAE uses the evaluated target's range for that frequency/split. Ordinary-test and held-out denominators can differ. Normalized tolerance rates are distinct from absolute-unit errors.
- Graph adjacency contains self-loops and is passed directly to `GCNConv`; no explicit normalization is applied in the examined graph path, unlike the normalized-adjacency description in the paper. The LNA graph builder compares width against raw `4` after adapter standardization; width distinctions can collapse in the graph branch. Dense/CCI inputs and the hybrid's auxiliary width input follow separate paths. Performance effects are not established by this code review.
- Early stopping uses validation loss. Separately, saved-weight retention can use ordinary-test MAE; immediate evaluation after training uses the current in-memory model. Do not interpret test-based retained checkpoints as independently validation-selected artifacts.
- `training_time_sec` measures fitting, while `loop_time_sec` includes other experiment work. Inference latency includes `model.predict` overhead, with two warm-up calls and ten timed repetitions by default, recorded separately for single-sample and batch predictions and ordinary/held-out inputs. Hardware-specific paper timings are not portable performance guarantees.
- One configured repetition/seed does not establish variability across independent random seeds. Saved matching summaries support numerical traceability but do not replace exact historical execution records or independent replication.

## Portability findings: future changes only

Only documentation and the missing PyTorch dependency declaration were updated; Python code was not changed. The present code can be used with prepared, schema-compatible data through its existing configuration/API. It is not yet a general drop-in framework for arbitrary circuit datasets.

Possible future improvements requiring separate authorization are: optionalizing the unused PyG import to remove the extra PyTorch installation; making the GPU diagnostic nonfatal; discovering frequencies from user data; decoupling conversion from private archive conventions; configuring arbitrary input/output roots; validating schemas and empty frequency slices early; avoiding diagnostic plotting in the core data path; and recording complete environment/code provenance for each run. Modeling corrections would be separate experiments, not silently substituted historical results.

`dataset/` and raw `Files/` inputs are not covered by the current `.gitignore`. Keep private data outside staged changes and inspect `git status --short` and staged filenames before publishing. `.gitignore` was intentionally preserved. No dataset, weights, or reference predictions are distributed, and publication of this code does not grant access to the private PDK/dataset.

## Repository layout

- `code/main.py`, `code/config.py`: main workflow and configuration.
- `code/run_experiment.py`, `code/experiments/`: existing named held-out presets.
- `code/data/`: loaders, preprocessors, adapters, and diagnostics.
- `code/dataset/`: Python topology metadata only; no circuit data.
- `code/ds_conv.py`: source-specific dataset conversion utilities.
- `code/models/`: FCNN, CCI-NN, GNN, and TARGet definitions.
- `code/training/`: fitting, evaluation, recording, and persistence.
- `code/analysis/`: plots, tables, provenance and runtime reports.
- `code/utils/`: paths, I/O, metrics, plotting, and seeds.

## License

Released under the MIT License. See [LICENSE](LICENSE).

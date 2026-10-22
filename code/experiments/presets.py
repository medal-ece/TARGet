"""Named experiment presets for reproducible, Run-button-friendly batches."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

from config import Config


@dataclass(frozen=True)
class ExperimentPreset:
    description: str
    targets: tuple[str, ...]
    output_tag: str
    overrides: dict[str, object]


_LNA_HELDOUT_DEFAULTS = {
    "workflow": "main",
    "main_arch": "LNA",
    "selected_topologies": [0, 1],
    "model_types": ["FCN", "CCI", "GNN", "CCIGNN-Sep"],
    "utilization": [1.0],
    "repetition": 1,
    "freq_mode": "uniform",
    "dataset_variant": None,
    "dataset_split": "main",
    "heldout_dataset_split": "heldout",
    "run_heldout_eval": True,
    "use_dataset_cache": True,
    "rebuild_dataset_cache": False,
    "run_analysis_after_experiment": True,
    "measure_inference_latency": True,
}


def _heldout_overrides(dataset_version: str, freq_count: int) -> dict[str, object]:
    return {
        **_LNA_HELDOUT_DEFAULTS,
        "dataset_version": dataset_version,
        "freq_count": freq_count,
    }


PRESETS: dict[str, ExperimentPreset] = {
    "lna_heldout_primary_10freq": ExperimentPreset(
        description="Primary LNA topology-heldout evaluation at 10 uniform frequencies.",
        targets=("GA", "GP", "NF"),
        output_tag="heldout_primary10",
        overrides=_heldout_overrides("v2_inf_cap_ind", 10),
    ),
    "lna_heldout_auxiliary_10freq": ExperimentPreset(
        description="Auxiliary LNA topology-heldout evaluation at 10 uniform frequencies.",
        targets=("GA", "GP", "NF"),
        output_tag="heldout_auxiliary10",
        overrides=_heldout_overrides("v2_inf_ind_inf", 10),
    ),
    "lna_heldout_v2_40freq": ExperimentPreset(
        description="V2 LNA topology-heldout evaluation at 40 uniform frequencies.",
        targets=("GA", "GP", "NF"),
        output_tag="heldout_v2_40",
        overrides=_heldout_overrides("v2_heldout", 40),
    ),
}


def get_preset(name: str) -> ExperimentPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown experiment preset '{name}'. Choose one of: {choices}") from exc


def build_preset_config(
    preset_name: str,
    target: str,
    model_types: list[str] | tuple[str, ...] | None = None,
) -> Config:
    """Build an independent Config for one target in a named preset."""
    preset = get_preset(preset_name)
    if target not in preset.targets:
        choices = ", ".join(preset.targets)
        raise ValueError(f"Target '{target}' is not part of '{preset_name}'. Choose one of: {choices}")

    config = Config()
    for field_name, value in preset.overrides.items():
        if not hasattr(config, field_name):
            raise AttributeError(f"Preset '{preset_name}' references unknown Config field '{field_name}'.")
        setattr(config, field_name, copy.deepcopy(value))
    config.targets = [target]
    if model_types is not None:
        if not model_types:
            raise ValueError("model_types cannot be empty.")
        config.model_types = list(model_types)
    return config


def run_preset(
    preset_name: str,
    targets: list[str] | tuple[str, ...] | None = None,
    model_types: list[str] | tuple[str, ...] | None = None,
    experiment_id: str | None = None,
) -> None:
    """Run all or selected targets from a preset.

    Supplying an existing experiment_id and a target/model subset supports
    resuming an interrupted batch without a dedicated resume script.
    """
    from main import main

    preset = get_preset(preset_name)
    selected_targets = tuple(targets) if targets is not None else preset.targets
    if not selected_targets:
        raise ValueError("targets cannot be empty.")

    for target in selected_targets:
        config = build_preset_config(preset_name, target, model_types=model_types)
        run_id = experiment_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{preset.output_tag}"
        print(f"\n\n=== {preset.description} Target: {target} ===")
        main(config=config, experiment_id=run_id)

"""Run a named experiment preset from VS Code's Run button.

Edit only the controls below. To resume an interrupted run, select the
remaining targets/models and set EXPERIMENT_ID to the original run id.
"""

from experiments.presets import run_preset


PRESET_NAME = "lna_heldout_primary_10freq"

# None runs every target/model defined by the preset.
TARGETS = None
MODEL_TYPES = None

# Keep None for a new run. Set the original id when resuming, for example:
# TARGETS = ["NF"]
# MODEL_TYPES = ["CCIGNN-Sep"]
# EXPERIMENT_ID = "20260619_072913_rebuttal40"
EXPERIMENT_ID = None


if __name__ == "__main__":
    run_preset(
        PRESET_NAME,
        targets=TARGETS,
        model_types=MODEL_TYPES,
        experiment_id=EXPERIMENT_ID,
    )

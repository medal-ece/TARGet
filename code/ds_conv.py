# In this file, we only convert the old datasets to the new format (with shuffling, dropping bad samples and so on).
import argparse
import json
import os
import re
import sys
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config
from dataset.topology_registry import describe_lna_match_selection
from models.factory import get_topology
from data.preprocessors import df_info_and_type_convert
from data.preprocessors import preprocess_dataset
from data.loaders import load_and_combine_topology_csvs, get_architecture_spec
from data.diagnostics import run_pca_diagnostics
from utils.file_io import csv_writer, split_and_write_dataset, write_metadata

def get_all_subsets(nums):
    result = []
    def backtrack(start, current_path):
        # Every time we enter the function, the 'current_path' is a valid subset
        if current_path:
            result.append(list(current_path))
        for i in range(start, len(nums)):
            current_path.append(nums[i])    # "Choose" the element
            backtrack(i + 1, current_path)  # "Explore" further
            current_path.pop()              # "Un-choose" (backtrack)
    backtrack(0, [])
    # Sort by length to match your specific order requirement
    return sorted(result, key=len)

def _dataset_name(topologies, selected_topologies):
    return (
        topologies[selected_topologies[0]]
        if len(selected_topologies) == 1
        else "combo_" + "_".join(map(str, selected_topologies))
    )


def parse_lna_modeling_maps(modeling_path):
    """Parse in_match/out_math ID maps from the old LNA modeling report."""
    maps = {"in_match": {}, "out_match": {}}
    if not modeling_path or not os.path.exists(modeling_path):
        return maps

    current_side = None
    current_class = None
    entry_re = re.compile(r"^\s*(\d+)\s*========>\s*(.+?)\s*$")
    with open(modeling_path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped == "in_match":
                current_side = "in_match"
                current_class = None
                continue
            if stripped in {"out_math", "out_match"}:
                current_side = "out_match"
                current_class = None
                continue
            if current_side is None:
                continue
            if ":" in stripped and "========>" not in stripped:
                # The mapping blocks have ended and experiment logs resumed.
                current_side = None
                current_class = None
                continue
            entry = entry_re.match(raw_line)
            if entry:
                if current_class is None:
                    continue
                topology_id = int(entry.group(1))
                maps[current_side][topology_id] = {
                    "class": current_class,
                    "parameters": entry.group(2),
                }
            elif "========>" not in stripped:
                current_class = stripped
    return maps


def _match_map_metadata(root, family_name="Cascode"):
    modeling_path = os.path.join(root, os.pardir, "Files", "LNA", "Results", family_name, "modeling.txt")
    maps = parse_lna_modeling_maps(modeling_path)
    return {
        "source_modeling_file": os.path.relpath(modeling_path, os.path.join(root, os.pardir)),
        "mapping_family_source": family_name,
        "note": (
            "CS and Cascode raw CSV numbering are treated as sharing the same in/out matching-network "
            "ID structure. Input and output maps are intentionally separate because the same numeric ID "
            "can belong to different classes on each side."
        ),
        "in_match": maps.get("in_match", {}),
        "out_match": maps.get("out_match", {}),
    }


def convert_dataset_subset(
    main_arch,
    selected_topologies,
    dataset_root_parts,
    source_topologies=None,
    output_dataset_name=None,
    include_file_ids=None,
    preprocessing=True,
    write=True,
    write_split=True,
    test_size=0.2,
    random_state=42,
    topology_col="topology_index",
    bound_check=False,
    run_pca=True,
    metadata_extra=None,
    add_lna_structural_ids=False,
):
    topologies, feature_list, spec_list = get_topology(main_arch)
    original_input_cols, original_output_cols, to_drop, desired_col_order = get_architecture_spec(main_arch)

    print("\nRunning for", selected_topologies, "\n")
    source_topologies = selected_topologies if source_topologies is None else source_topologies
    main_topology = output_dataset_name or _dataset_name(topologies, selected_topologies)

    print("[INFO] Architecture:", main_arch)
    print("[INFO] Selected topology indices:", selected_topologies)
    print("[INFO] Source topology indices:", source_topologies)
    print("[INFO] Output dataset name:", main_topology)
    if include_file_ids is not None:
        print("[INFO] Included raw file IDs:", f"{min(include_file_ids)}..{max(include_file_ids)}", f"({len(include_file_ids)} IDs)")

    root = os.path.abspath(os.path.dirname(__file__))
    files_root = os.path.join(root, os.pardir, "Files", main_arch, "CSV", "main", "results")
    output_path = os.path.join(root, os.pardir, "dataset", *dataset_root_parts, main_topology)
    images_path = os.path.join(root, os.pardir, "images")
    os.makedirs(output_path, exist_ok=True)

    df_origin = load_and_combine_topology_csvs(
        files_root=files_root,
        topologies=topologies,
        selected_topologies=source_topologies,
        topology_col=topology_col,
        include_file_ids=set(include_file_ids) if include_file_ids is not None else None,
        add_lna_structural_ids=add_lna_structural_ids,
    )

    structural_cols = [
        col for col in ["in_topology_id", "out_topology_id"]
        if col in df_origin.columns and col not in desired_col_order
    ]
    df = df_origin[desired_col_order + structural_cols + [topology_col]]

    META_COLS = {"Freq", "freq", "s2", "w", "in_topology_id", "out_topology_id", topology_col}
    num_cols = [
        c for c in df.columns
        if c not in META_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    df_noinf = None
    df_no_nan_step_3 = None
    if preprocessing:
        df_type_convert = df_info_and_type_convert(df)
        df_noinf, df_no_nan_step_2, df_no_nan_step_3, df_outliers = preprocess_dataset(
            df_type_convert,
            num_cols=num_cols,
            to_drop=to_drop,
            topology_col=topology_col,
            random_state=random_state,
        )

        df = df_no_nan_step_2.copy()
        if bound_check:
            invalid = pd.Series(False, index=df.index)
            numeric_bounds = {}
            for col, (lo, hi) in numeric_bounds.items():
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce")
                    invalid |= (s < lo) | (s > hi) | s.isna()
            df = df[~invalid]

        INPUT_COLS = [c for c in original_input_cols if c in df.columns]
        spec_list = [t for t in spec_list if t in df.columns]

        X = df[INPUT_COLS].copy()
        Y = df[spec_list].copy()

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

        too_missing = X_train.columns[X_train.isna().mean() > 0.40]
        X_train = X_train.drop(columns=too_missing)
        X_test = X_test.drop(columns=[c for c in too_missing if c in X_test], errors="ignore")

        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

    if run_pca and df_noinf is not None and df_no_nan_step_3 is not None:
        run_pca_diagnostics(
            df_noinf=df_no_nan_step_3,
            feature_list=feature_list,
            images_path=images_path,
            main_arch=main_arch,
            dataset_name=main_topology,
        )

    if write_split:
        train_df, test_df = split_and_write_dataset(df=df, output_path=output_path, test_size=test_size, write=write)
    else:
        train_df = pd.DataFrame(columns=df.columns)
        test_df = df.copy()
        if write:
            from utils.file_io import csv_writer

            print("[INFO] Writing HELDOUT dataset chunks as test_*.csv")
            csv_writer(test_df, output_path, prefix="test")

    if write:
        write_metadata(
            output_path=output_path,
            main_arch=main_arch,
            dataset_name=main_topology,
            selected_topologies=selected_topologies,
            df_origin=df_origin,
            extra=metadata_extra,
        )
    return train_df, test_df


def convert_standard_dataset(main_arch="LNA", write=True, preprocessing=True, run_pca=True):
    topologies_nums = [0, 1] if main_arch == "LNA" else [0, 1, 2, 3]
    topologies_subnets = get_all_subsets(topologies_nums)
    for selected_topologies in topologies_subnets:
        convert_dataset_subset(
            main_arch=main_arch,
            selected_topologies=selected_topologies,
            dataset_root_parts=["v1", main_arch, "main"],
            preprocessing=preprocessing,
            write=write,
            write_split=True,
            run_pca=run_pca,
            metadata_extra={
                "dataset_version": "v1",
                "dataset_split": "main",
                "raw_file_id_policy": "original converted dataset without heldout ID separation",
            },
        )


def convert_lna_heldout_v2_raw_id(write=True, preprocessing=True, run_pca=False):
    main_arch = "LNA"
    topologies_nums = [0, 1]
    topologies_subnets = get_all_subsets(topologies_nums)
    main_ids = set(range(1, 61))
    heldout_ids = set(range(61, 65))

    for selected_topologies in topologies_subnets:
        convert_dataset_subset(
            main_arch=main_arch,
            selected_topologies=selected_topologies,
            dataset_root_parts=["v2_heldout", main_arch, "main"],
            include_file_ids=main_ids,
            preprocessing=preprocessing,
            write=write,
            write_split=True,
            run_pca=run_pca,
            metadata_extra={
                "dataset_version": "v2_heldout",
                "dataset_split": "main",
                "raw_file_id_policy": "include IDs 1-60; heldout IDs 61-64 excluded before train/test split",
                "included_raw_file_ids": [1, 60],
                "heldout_raw_file_ids": [61, 64],
            },
        )
        convert_dataset_subset(
            main_arch=main_arch,
            selected_topologies=selected_topologies,
            dataset_root_parts=["v2_heldout", main_arch, "heldout"],
            include_file_ids=heldout_ids,
            preprocessing=preprocessing,
            write=write,
            write_split=False,
            run_pca=False,
            metadata_extra={
                "dataset_version": "v2_heldout",
                "dataset_split": "heldout",
                "raw_file_id_policy": "zero-shot heldout only; IDs 61-64 are never used for training, validation, preprocessing-stat fitting, or model selection",
                "included_raw_file_ids": [61, 64],
                "training_raw_file_ids": [1, 60],
                "evaluation_only": True,
            },
        )


def convert_lna_cross_family_v3_raw_family(write=True, preprocessing=True, run_pca=False):
    main_arch = "LNA"
    model_topologies = [0, 1]
    output_dataset_name = "combo_0_1"
    variants = [
        {
            "name": "CS_to_Cascode",
            "train_source_topologies": [0],
            "heldout_source_topologies": [1],
            "description": "train on CS only; zero-shot evaluate on Cascode",
        },
        {
            "name": "Cascode_to_CS",
            "train_source_topologies": [1],
            "heldout_source_topologies": [0],
            "description": "train on Cascode only; zero-shot evaluate on CS",
        },
    ]

    for variant in variants:
        convert_dataset_subset(
            main_arch=main_arch,
            selected_topologies=model_topologies,
            source_topologies=variant["train_source_topologies"],
            output_dataset_name=output_dataset_name,
            dataset_root_parts=["v3_cross_family", variant["name"], main_arch, "main"],
            preprocessing=preprocessing,
            write=write,
            write_split=True,
            run_pca=run_pca,
            metadata_extra={
                "dataset_version": "v3_cross_family",
                "dataset_variant": variant["name"],
                "dataset_split": "main",
                "raw_file_id_policy": "cross-family zero-shot training split; all available raw IDs from source family",
                "zero_shot_policy": variant["description"],
                "model_topology_indices": model_topologies,
                "source_topology_indices": variant["train_source_topologies"],
                "heldout_source_topology_indices": variant["heldout_source_topologies"],
            },
        )
        convert_dataset_subset(
            main_arch=main_arch,
            selected_topologies=model_topologies,
            source_topologies=variant["heldout_source_topologies"],
            output_dataset_name=output_dataset_name,
            dataset_root_parts=["v4_cross_family_heldout", variant["name"], main_arch, "heldout"],
            include_file_ids=heldout_ids,
            preprocessing=preprocessing,
            write=write,
            write_split=False,
            run_pca=False,
            metadata_extra={
                "dataset_version": "v4_cross_family_heldout",
                "dataset_variant": variant["name"],
                "dataset_split": "heldout",
                "raw_file_id_policy": "cross-family heldout evaluation split; unseen-family raw IDs 61-64 only",
                "zero_shot_policy": variant["description"],
                "evaluation_only": True,
                "model_topology_indices": model_topologies,
                "source_topology_indices": variant["heldout_source_topologies"],
                "included_raw_file_ids": [61, 64],
                "training_source_topology_indices": variant["train_source_topologies"],
                "training_raw_file_ids": [1, 60],
            },
        )
        convert_dataset_subset(
            main_arch=main_arch,
            selected_topologies=model_topologies,
            source_topologies=variant["heldout_source_topologies"],
            output_dataset_name=output_dataset_name,
            dataset_root_parts=["v3_cross_family", variant["name"], main_arch, "heldout"],
            preprocessing=preprocessing,
            write=write,
            write_split=False,
            run_pca=False,
            metadata_extra={
                "dataset_version": "v3_cross_family",
                "dataset_variant": variant["name"],
                "dataset_split": "heldout",
                "raw_file_id_policy": "cross-family zero-shot heldout split; all available raw IDs from unseen family",
                "zero_shot_policy": variant["description"],
                "evaluation_only": True,
                "model_topology_indices": model_topologies,
                "source_topology_indices": variant["heldout_source_topologies"],
                "training_source_topology_indices": variant["train_source_topologies"],
            },
        )


def convert_lna_cross_family_heldout_v4_raw_id(write=True, preprocessing=True, run_pca=False):
    main_arch = "LNA"
    model_topologies = [0, 1]
    output_dataset_name = "combo_0_1"
    main_ids = set(range(1, 61))
    heldout_ids = set(range(61, 65))
    variants = [
        {
            "name": "CS_to_Cascode",
            "train_source_topologies": [0],
            "heldout_source_topologies": [1],
            "description": "train on CS IDs 1-60; zero-shot evaluate on Cascode IDs 61-64",
        },
        {
            "name": "Cascode_to_CS",
            "train_source_topologies": [1],
            "heldout_source_topologies": [0],
            "description": "train on Cascode IDs 1-60; zero-shot evaluate on CS IDs 61-64",
        },
    ]

    for variant in variants:
        convert_dataset_subset(
            main_arch=main_arch,
            selected_topologies=model_topologies,
            source_topologies=variant["train_source_topologies"],
            output_dataset_name=output_dataset_name,
            dataset_root_parts=["v4_cross_family_heldout", variant["name"], main_arch, "main"],
            include_file_ids=main_ids,
            preprocessing=preprocessing,
            write=write,
            write_split=True,
            run_pca=run_pca,
            metadata_extra={
                "dataset_version": "v4_cross_family_heldout",
                "dataset_variant": variant["name"],
                "dataset_split": "main",
                "raw_file_id_policy": "cross-family heldout training split; source-family raw IDs 1-60 only",
                "zero_shot_policy": variant["description"],
                "model_topology_indices": model_topologies,
                "source_topology_indices": variant["train_source_topologies"],
                "included_raw_file_ids": [1, 60],
                "heldout_source_topology_indices": variant["heldout_source_topologies"],
                "heldout_raw_file_ids": [61, 64],
            },
        )


def convert_lna_structured_v5(write=True, preprocessing=True, run_pca=False):
    """Regenerate the original LNA dataset with explicit in/out matching-network IDs."""
    main_arch = "LNA"
    topologies_nums = [0, 1]
    topologies_subnets = get_all_subsets(topologies_nums)
    root = os.path.abspath(os.path.dirname(__file__))
    match_maps = _match_map_metadata(root, family_name="Cascode")

    for selected_topologies in topologies_subnets:
        convert_dataset_subset(
            main_arch=main_arch,
            selected_topologies=selected_topologies,
            dataset_root_parts=["v5", main_arch, "main"],
            preprocessing=preprocessing,
            write=write,
            write_split=True,
            run_pca=run_pca,
            add_lna_structural_ids=True,
            metadata_extra={
                "dataset_version": "v5",
                "dataset_split": "main",
                "raw_file_id_policy": (
                    "structured original LNA conversion; each row carries explicit input and output "
                    "matching-network IDs. Filename wmain_<width>_in_<id>.csv supplies width and "
                    "input ID; row order within each frequency block supplies output ID 1..64."
                ),
                "match_id_columns": {
                    "width": "w",
                    "input_topology_id": "in_topology_id",
                    "output_topology_id": "out_topology_id",
                },
                "output_id_inference": "groupby(freq).cumcount()+1 within each old all/wmain_*_in_*.csv file",
                "match_topology_maps": match_maps,
            },
        )


def _read_dataset_chunks(dataset_dir):
    files = sorted(
        [
            os.path.join(dataset_dir, name)
            for name in os.listdir(dataset_dir)
            if name.endswith(".csv") and (name.startswith("train_") or name.startswith("test_"))
        ]
    )
    if not files:
        raise FileNotFoundError(f"No train/test CSV chunks found under {dataset_dir}")
    print(f"[INFO] Reading {len(files)} v5 chunks from {dataset_dir}")
    return pd.concat((pd.read_csv(path) for path in files), ignore_index=True)


def _prepare_generated_dataset_dir(output_path):
    os.makedirs(output_path, exist_ok=True)
    for name in os.listdir(output_path):
        path = os.path.join(output_path, name)
        if name.startswith(("train_", "test_")) and name.endswith(".csv"):
            os.remove(path)
        elif name == "metadata.json":
            os.remove(path)
        elif name == ".cache" and os.path.isdir(path):
            shutil.rmtree(path)


def _lna_heldout_mask(df, in_ids, out_ids, rule):
    in_mask = df["in_topology_id"].isin(in_ids)
    out_mask = df["out_topology_id"].isin(out_ids)
    if rule == "either_side":
        return in_mask | out_mask
    if rule == "both_sides":
        return in_mask & out_mask
    if rule == "input_only":
        return in_mask
    if rule == "output_only":
        return out_mask
    raise ValueError(f"Unknown lna_heldout_match_rule: {rule}")


def _convert_lna_heldout_from_v5(
    *,
    output_dataset_version,
    in_topology_selections,
    out_topology_selections,
    rule,
    write=True,
    test_size=0.2,
    policy_note=None,
    topology_sets=None,
):
    """Derive a topology-aware v2 heldout dataset from v5."""
    main_arch = "LNA"
    topologies, _, _ = get_topology(main_arch)
    topology_sets = get_all_subsets([0, 1]) if topology_sets is None else topology_sets
    root = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(root, os.pardir))
    selection = describe_lna_match_selection(
        in_topology_selections,
        out_topology_selections,
    )
    in_ids = selection["in_topology_ids"]
    out_ids = selection["out_topology_ids"]

    print(f"[INFO] {output_dataset_version} heldout selection:")
    print(json.dumps({**selection, "rule": rule, "policy_note": policy_note}, indent=2))

    for selected_topologies in topology_sets:
        dataset_name = _dataset_name(topologies, selected_topologies)
        source_dir = os.path.join(project_root, "dataset", "v5", main_arch, "main", dataset_name)
        df = _read_dataset_chunks(source_dir)
        required = {"in_topology_id", "out_topology_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Source v5 dataset {source_dir} is missing required columns: {sorted(missing)}")

        heldout_mask = _lna_heldout_mask(df, in_ids=in_ids, out_ids=out_ids, rule=rule)
        main_df = df.loc[~heldout_mask].sample(frac=1, random_state=42).reset_index(drop=True)
        heldout_df = df.loc[heldout_mask].sample(frac=1, random_state=42).reset_index(drop=True)

        main_output = os.path.join(project_root, "dataset", output_dataset_version, main_arch, "main", dataset_name)
        heldout_output = os.path.join(project_root, "dataset", output_dataset_version, main_arch, "heldout", dataset_name)
        if write:
            _prepare_generated_dataset_dir(main_output)
            _prepare_generated_dataset_dir(heldout_output)

        print(f"\n[INFO] {dataset_name}: source={len(df):,}, main={len(main_df):,}, heldout={len(heldout_df):,}")
        train_df, test_df = split_and_write_dataset(main_df, main_output, test_size=test_size, write=write)
        if write:
            print("[INFO] Writing HELDOUT dataset chunks")
            csv_writer(heldout_df, heldout_output, prefix="test")

        metadata_extra = {
            "dataset_version": output_dataset_version,
            "dataset_split": "main",
            "source_dataset_version": "v5",
            "heldout_policy": "topology-aware split derived from v5 using explicit in_topology_id and out_topology_id",
            "policy_note": policy_note,
            "heldout_selection": {**selection, "rule": rule},
            "source_rows": int(len(df)),
            "main_rows": int(len(main_df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "heldout_rows": int(len(heldout_df)),
        }
        if write:
            write_metadata(
                output_path=main_output,
                main_arch=main_arch,
                dataset_name=dataset_name,
                selected_topologies=selected_topologies,
                df_origin=main_df,
                extra=metadata_extra,
            )
            heldout_meta = metadata_extra.copy()
            heldout_meta["dataset_split"] = "heldout"
            write_metadata(
                output_path=heldout_output,
                main_arch=main_arch,
                dataset_name=dataset_name,
                selected_topologies=selected_topologies,
                df_origin=heldout_df,
                extra=heldout_meta,
            )


def convert_lna_heldout_v2(write=True, test_size=0.2):
    """Derive the default topology-aware v2 heldout dataset from v5."""
    config = Config()
    _convert_lna_heldout_from_v5(
        output_dataset_version="v2_heldout",
        in_topology_selections=config.lna_heldout_in_match_topologies,
        out_topology_selections=config.lna_heldout_out_match_topologies,
        rule=config.lna_heldout_match_rule,
        write=write,
        test_size=test_size,
        policy_note="Config-driven v2 heldout split.",
    )


def convert_lna_heldout_v2_inf_ind_inf(write=True, test_size=0.2):
    """Derive a v2 heldout split using only inf_ind_inf (ID 64) on input/output."""
    _convert_lna_heldout_from_v5(
        output_dataset_version="v2_inf_ind_inf",
        in_topology_selections=["inf_ind_inf"],
        out_topology_selections=["inf_ind_inf"],
        rule="either_side",
        write=write,
        test_size=test_size,
        policy_note=(
            "Supervisor-requested zero-shot split: remove only matching-network ID 64 "
            "(inf_ind_inf) from input and output sides for heldout evaluation."
        ),
    )


def convert_lna_heldout_v2_inf_cap_ind(write=True, test_size=0.2, combo_only=False):
    """Derive a v2 heldout split using inf_cap_ind on input/output."""
    _convert_lna_heldout_from_v5(
        output_dataset_version="v2_inf_cap_ind",
        in_topology_selections=["inf_cap_ind"],
        out_topology_selections=["inf_cap_ind"],
        rule="either_side",
        write=write,
        test_size=test_size,
        policy_note=(
            "Topology-aware zero-shot split: remove input inf_cap_ind IDs 61-63 "
            "and output inf_cap_ind IDs 57-59 for heldout evaluation."
        ),
        topology_sets=[[0, 1]] if combo_only else None,
    )


def _family_dataset_name(family_index):
    topologies, _, _ = get_topology("LNA")
    return topologies[family_index]


def _read_v5_family(project_root, family_index):
    family_name = _family_dataset_name(family_index)
    source_dir = os.path.join(project_root, "dataset", "v5", "LNA", "main", family_name)
    df = _read_dataset_chunks(source_dir)
    required = {"w", "in_topology_id", "out_topology_id", "topology_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source v5 dataset {source_dir} is missing required columns: {sorted(missing)}")
    return df


def _write_family_split(
    *,
    output_root,
    dataset_version,
    dataset_variant,
    dataset_split,
    dataset_name,
    selected_topologies,
    df,
    write,
    write_split,
    test_size,
    metadata_extra,
):
    if write:
        _prepare_generated_dataset_dir(output_root)
    if write_split:
        train_df, test_df = split_and_write_dataset(df, output_root, test_size=test_size, write=write)
    else:
        train_df = pd.DataFrame(columns=df.columns)
        test_df = df.copy()
        if write:
            print("[INFO] Writing HELDOUT dataset chunks")
            csv_writer(test_df, output_root, prefix="test")

    if write:
        write_metadata(
            output_path=output_root,
            main_arch="LNA",
            dataset_name=dataset_name,
            selected_topologies=selected_topologies,
            df_origin=df,
            extra={
                "dataset_version": dataset_version,
                "dataset_variant": dataset_variant,
                "dataset_split": dataset_split,
                "source_dataset_version": "v5",
                "source_dataset_policy": "structured LNA rows with explicit w, in_topology_id, and out_topology_id",
                "source_rows": int(len(df)),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                **metadata_extra,
            },
        )
    return train_df, test_df


def convert_lna_cross_family_v3(write=True, test_size=0.2):
    """Derive cross-family datasets from v5."""
    main_arch = "LNA"
    model_topologies = [0, 1]
    output_dataset_name = "combo_0_1"
    root = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(root, os.pardir))
    variants = [
        {
            "name": "CS_to_Cascode",
            "train_family": 0,
            "heldout_family": 1,
            "description": "train/test on all CS rows; zero-shot evaluate on all Cascode rows",
        },
        {
            "name": "Cascode_to_CS",
            "train_family": 1,
            "heldout_family": 0,
            "description": "train/test on all Cascode rows; zero-shot evaluate on all CS rows",
        },
    ]

    for variant in variants:
        train_df = _read_v5_family(project_root, variant["train_family"]).sample(frac=1, random_state=42).reset_index(drop=True)
        heldout_df = _read_v5_family(project_root, variant["heldout_family"]).sample(frac=1, random_state=42).reset_index(drop=True)
        train_family_name = _family_dataset_name(variant["train_family"])
        heldout_family_name = _family_dataset_name(variant["heldout_family"])
        print(f"\n[INFO] v3 {variant['name']}: main={len(train_df):,}, heldout={len(heldout_df):,}")

        main_output = os.path.join(project_root, "dataset", "v3_cross_family", variant["name"], main_arch, "main", output_dataset_name)
        heldout_output = os.path.join(project_root, "dataset", "v3_cross_family", variant["name"], main_arch, "heldout", output_dataset_name)

        _write_family_split(
            output_root=main_output,
            dataset_version="v3_cross_family",
            dataset_variant=variant["name"],
            dataset_split="main",
            dataset_name=output_dataset_name,
            selected_topologies=model_topologies,
            df=train_df,
            write=write,
            write_split=True,
            test_size=test_size,
            metadata_extra={
                "zero_shot_policy": variant["description"],
                "training_family": train_family_name,
                "heldout_family": heldout_family_name,
                "family_shift": True,
                "topology_holdout": False,
            },
        )
        _write_family_split(
            output_root=heldout_output,
            dataset_version="v3_cross_family",
            dataset_variant=variant["name"],
            dataset_split="heldout",
            dataset_name=output_dataset_name,
            selected_topologies=model_topologies,
            df=heldout_df,
            write=write,
            write_split=False,
            test_size=test_size,
            metadata_extra={
                "zero_shot_policy": variant["description"],
                "training_family": train_family_name,
                "heldout_family": heldout_family_name,
                "family_shift": True,
                "topology_holdout": False,
                "evaluation_only": True,
            },
        )


def convert_lna_cross_family_heldout_v4(write=True, test_size=0.2):
    """Derive cross-family plus matching-topology heldout datasets from v5."""
    config = Config()
    main_arch = "LNA"
    model_topologies = [0, 1]
    output_dataset_name = "combo_0_1"
    root = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(root, os.pardir))
    selection = describe_lna_match_selection(
        config.lna_heldout_in_match_topologies,
        config.lna_heldout_out_match_topologies,
    )
    in_ids = selection["in_topology_ids"]
    out_ids = selection["out_topology_ids"]
    rule = config.lna_heldout_match_rule
    variants = [
        {
            "name": "CS_to_Cascode",
            "train_family": 0,
            "heldout_family": 1,
            "description": "train/test on CS excluding selected match topologies; zero-shot evaluate on selected Cascode match topologies",
        },
        {
            "name": "Cascode_to_CS",
            "train_family": 1,
            "heldout_family": 0,
            "description": "train/test on Cascode excluding selected match topologies; zero-shot evaluate on selected CS match topologies",
        },
    ]

    print("[INFO] V4 heldout selection:")
    print(json.dumps({**selection, "rule": rule}, indent=2))

    for variant in variants:
        source_df = _read_v5_family(project_root, variant["train_family"])
        target_df = _read_v5_family(project_root, variant["heldout_family"])
        source_mask = _lna_heldout_mask(source_df, in_ids=in_ids, out_ids=out_ids, rule=rule)
        target_mask = _lna_heldout_mask(target_df, in_ids=in_ids, out_ids=out_ids, rule=rule)
        main_df = source_df.loc[~source_mask].sample(frac=1, random_state=42).reset_index(drop=True)
        heldout_df = target_df.loc[target_mask].sample(frac=1, random_state=42).reset_index(drop=True)
        train_family_name = _family_dataset_name(variant["train_family"])
        heldout_family_name = _family_dataset_name(variant["heldout_family"])
        print(f"\n[INFO] v4 {variant['name']}: main={len(main_df):,}, heldout={len(heldout_df):,}")

        main_output = os.path.join(project_root, "dataset", "v4_cross_family_heldout", variant["name"], main_arch, "main", output_dataset_name)
        heldout_output = os.path.join(project_root, "dataset", "v4_cross_family_heldout", variant["name"], main_arch, "heldout", output_dataset_name)

        common_meta = {
            "heldout_selection": {**selection, "rule": rule},
            "zero_shot_policy": variant["description"],
            "training_family": train_family_name,
            "heldout_family": heldout_family_name,
            "family_shift": True,
            "topology_holdout": True,
        }
        _write_family_split(
            output_root=main_output,
            dataset_version="v4_cross_family_heldout",
            dataset_variant=variant["name"],
            dataset_split="main",
            dataset_name=output_dataset_name,
            selected_topologies=model_topologies,
            df=main_df,
            write=write,
            write_split=True,
            test_size=test_size,
            metadata_extra={
                **common_meta,
                "excluded_source_heldout_rows": int(source_mask.sum()),
            },
        )
        _write_family_split(
            output_root=heldout_output,
            dataset_version="v4_cross_family_heldout",
            dataset_variant=variant["name"],
            dataset_split="heldout",
            dataset_name=output_dataset_name,
            selected_topologies=model_topologies,
            df=heldout_df,
            write=write,
            write_split=False,
            test_size=test_size,
            metadata_extra={
                **common_meta,
                "target_nonheldout_rows": int((~target_mask).sum()),
                "evaluation_only": True,
            },
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "standard",
            "lna-heldout-v2",
            "lna-heldout-v2-inf-ind-inf",
            "lna-heldout-v2-inf-cap-ind",
            "lna-heldout-v2-inf-cap-ind-combo",
            "lna-cross-family-v3",
            "lna-cross-family-heldout-v4",
            "lna-structured-v5",
        ],
        default="lna-heldout-v2",
    )
    parser.add_argument("--arch", choices=["LNA", "phase_shifter"], default="LNA")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--run-pca", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    write = not args.no_write
    preprocessing = not args.no_preprocess
    if args.mode == "lna-heldout-v2":
        convert_lna_heldout_v2(write=write)
    elif args.mode == "lna-heldout-v2-inf-ind-inf":
        convert_lna_heldout_v2_inf_ind_inf(write=write)
    elif args.mode == "lna-heldout-v2-inf-cap-ind":
        convert_lna_heldout_v2_inf_cap_ind(write=write)
    elif args.mode == "lna-heldout-v2-inf-cap-ind-combo":
        convert_lna_heldout_v2_inf_cap_ind(write=write, combo_only=True)
    elif args.mode == "lna-cross-family-v3":
        convert_lna_cross_family_v3(write=write)
    elif args.mode == "lna-cross-family-heldout-v4":
        convert_lna_cross_family_heldout_v4(write=write)
    elif args.mode == "lna-structured-v5":
        convert_lna_structured_v5(write=write, preprocessing=preprocessing, run_pca=args.run_pca)
    else:
        convert_standard_dataset(main_arch=args.arch, write=write, preprocessing=preprocessing, run_pca=args.run_pca)


def legacy_main():
    # Configuration
    main_arch = "LNA" # LNA, phase_shifter
    topologies_nums = [0, 1]
    topologies_subnets = get_all_subsets(topologies_nums) # indices of topologies to include; e.g., [0], [1], [0,1], etc.
    preprocessing = True  # True to do preprocessing (drop duplicates, handle missing values, drop bad samples)
    write = True  # True to write the new dataset, False to just analyze
    test_size = 0.2  # 20% test, 80% train
    random_state = 42
    TOPOLOGY_COL = "topology_index"
    bound_check = False  # True to do hard-bound validity checking after preprocessing

    # Architecture specification
    # Get topology names and feature/output lists
    topologies, feature_list, spec_list = get_topology(main_arch)
    # Architecture-specific column spec
    original_input_cols, original_output_cols, to_drop, desired_col_order = get_architecture_spec(main_arch)

    for selected_topologies in topologies_subnets:
        print("\nRunning for", selected_topologies, "\n")
        # New Dataset naming (folder name)
        main_topology = (
            topologies[selected_topologies[0]]
            if len(selected_topologies) == 1
            else "combo_" + "_".join(map(str, selected_topologies))
        )
        # main_topology = topologies[0] if len(selected_topologies) == 1 else "combo_" + "_".join(str(i) for i in selected_topologies)

        print("[INFO] Architecture:", main_arch)
        print("[INFO] Selected topology indices:", selected_topologies)
        print("[INFO] Output dataset name:", main_topology)

        # Paths
        root = os.path.abspath(os.path.dirname(__file__))
        files_root = os.path.join(root, os.pardir, "Files", main_arch, "CSV", "main", "results")
        output_path = os.path.join(root, os.pardir, "dataset", main_arch, "main", main_topology)
        images_path = os.path.join(root, os.pardir, "images")
        os.makedirs(output_path, exist_ok=True)
        
        # Load and combine raw dataset
        df_origin = load_and_combine_topology_csvs(files_root=files_root, topologies=topologies, selected_topologies=selected_topologies, topology_col=TOPOLOGY_COL)

        # Column ordering
        df = df_origin[desired_col_order + [TOPOLOGY_COL]]

        # Numeric columns (exclude topology index)
        META_COLS = {"Freq", "s2", TOPOLOGY_COL}
        num_cols = [
            c for c in df.columns
            if c not in META_COLS
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        # Preprocessing
        if preprocessing:
            # ======= info =======
            df_type_convert = df_info_and_type_convert(df)
            df_noinf, df_no_nan_step_2, df_no_nan_step_3, df_outliers = preprocess_dataset(df_type_convert, num_cols=num_cols, to_drop=to_drop, topology_col=TOPOLOGY_COL, random_state=random_state)

            # Optional hard-bound validation
            df = df_no_nan_step_2.copy()
            if bound_check:
                # Validity rules — edit as needed
                # Bound checks for numeric columns
                invalid = pd.Series(False, index=df.index)  # Build a mask of invalid rows
                numeric_bounds = {}                         # e.g., {"prob": (0.0, 1.0), "age": (0, 120)}
                for col, (lo, hi) in numeric_bounds.items():
                    if col in df.columns:
                        s = pd.to_numeric(df[col], errors="coerce")
                        invalid |= (s < lo) | (s > hi) | s.isna()
                bad_rows = df[invalid]
                df = df[~invalid]

            # ======= select columns =======
            INPUT_COLS = [c for c in original_input_cols if c in df.columns]
            spec_list    = [t for t in spec_list if t in df.columns]

            X = df[INPUT_COLS].copy()
            Y = df[spec_list].copy()

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

            # 1) Handle missing values (numeric)
            # ------------------------------------
            # Drop columns with too many NaNs (train-based)
            too_missing = X_train.columns[X_train.isna().mean() > 0.40]
            X_train = X_train.drop(columns=too_missing)
            X_test  = X_test.drop(columns=[c for c in too_missing if c in X_test], errors="ignore")

            # Median impute using train medians
            medians = X_train.median()
            X_train = X_train.fillna(medians)
            X_test  = X_test.fillna(medians)
        
        # PCA diagnostics
        assert df_noinf is not None, "df_noinf is required for PCA diagnostics"
        run_pca_diagnostics(df_noinf=df_no_nan_step_3, feature_list=feature_list, images_path=images_path, main_arch=main_arch, dataset_name=main_topology,)
        
        # Train / test split and writing
        train_df, test_df = split_and_write_dataset(df=df, output_path=output_path, test_size=test_size, write=write)

        # Write metadata
        if write:
            write_metadata(output_path=output_path, main_arch=main_arch, dataset_name=main_topology, selected_topologies=selected_topologies, df_origin=df_origin)


if __name__ == "__main__":
    main()

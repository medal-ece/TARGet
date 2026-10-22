# dataset/topology_registry.py
"""Permanent topology ID registries for generated datasets."""

from __future__ import annotations


LNA_MATCH_TOPOLOGY_GROUPS = {
    "in": {
        "wire": [1],
        "cap_ind_cap": list(range(2, 16)),
        "cap_indcap_inf": list(range(16, 31)),
        "ind_cap_ind": list(range(31, 46)),
        "ind_indcap_inf": list(range(46, 61)),
        "inf_cap_ind": [61, 62, 63],
        "inf_ind_inf": [64],
    },
    "out": {
        "cap_ind_cap": list(range(1, 15)),
        "cap_indcap_inf": list(range(15, 29)),
        "ind_cap_ind": list(range(29, 43)),
        "ind_indcap_inf": list(range(43, 57)),
        "inf_cap_ind": [57, 58, 59],
        "inf_cap_ind_inf": [60, 61, 62],
        "inf_ind_inf_par": [63],
        "inf_ind_inf": [64],
    },
}


DEFAULT_LNA_HELDOUT_IN_MATCH_TOPOLOGIES = ["inf_cap_ind", "inf_ind_inf"]
DEFAULT_LNA_HELDOUT_OUT_MATCH_TOPOLOGIES = [
    "inf_cap_ind_inf",
    "inf_ind_inf_par",
    "inf_ind_inf",
]


def _normalize_side(side: str) -> str:
    side_key = str(side).strip().lower()
    aliases = {
        "input": "in",
        "in_match": "in",
        "in": "in",
        "output": "out",
        "out_match": "out",
        "out_math": "out",
        "out": "out",
    }
    if side_key not in aliases:
        raise ValueError(f"Unknown LNA matching side '{side}'. Expected 'in' or 'out'.")
    return aliases[side_key]


def available_lna_match_topologies(side: str) -> list[str]:
    """Return the registered topology group names for the requested side."""
    side_key = _normalize_side(side)
    return list(LNA_MATCH_TOPOLOGY_GROUPS[side_key].keys())


def resolve_lna_match_ids(side: str, selections: list[str | int]) -> list[int]:
    """Resolve topology group names and/or explicit IDs into sorted unique IDs."""
    side_key = _normalize_side(side)
    groups = LNA_MATCH_TOPOLOGY_GROUPS[side_key]
    ids: set[int] = set()
    unknown: list[str] = []

    for item in selections or []:
        if isinstance(item, int):
            ids.add(item)
            continue
        text = str(item).strip()
        if not text:
            continue
        if text.isdigit():
            ids.add(int(text))
        elif text in groups:
            ids.update(groups[text])
        else:
            unknown.append(text)

    if unknown:
        raise ValueError(
            f"Unknown LNA {side_key} topology group(s): {unknown}. "
            f"Available: {available_lna_match_topologies(side_key)}"
        )
    return sorted(ids)


def describe_lna_match_selection(in_selections: list[str | int], out_selections: list[str | int]) -> dict:
    """Return a serializable description of an LNA topology heldout choice."""
    return {
        "in_topology_groups": list(in_selections or []),
        "in_topology_ids": resolve_lna_match_ids("in", in_selections or []),
        "out_topology_groups": list(out_selections or []),
        "out_topology_ids": resolve_lna_match_ids("out", out_selections or []),
    }

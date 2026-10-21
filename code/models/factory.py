# code/models/factory.py
"""Model selection and construction helpers."""

from models.fcn import FC_net
from models.cci_nn import CCI_net
from models.gnn import GNN_net
from models.cci_gnn import CCIGNN_Sep_net

def get_topology(arch, cartesian=False):
    """Return topology names, input features, and targets for the selected architecture."""
    if arch == "phase_shifter":
        topologys = [
            'sw1_0_sw2_0', # '0',
            'sw1_0_sw2_1', # '1',
            'sw1_1_sw2_0', # '2',
            'sw1_1_sw2_1', # '3',
            # Combinations of topologies datasets
            # We don't have self combinations.
            # Also, there is no difference between _1_2 and _2_1.
            # 'combo_0_1',
            # 'combo_0_2',
            # 'combo_0_3',
            # 'combo_1_2',
            # 'combo_1_3',
            # 'combo_2_3',
            # 'combo_0_1_2',
            # 'combo_0_1_3',
            # 'combo_0_2_3',
            # 'combo_1_2_3',
            # 'combo_0_1_2_3'
        ]
        if cartesian:
            x_set = [
                's1_s11_x', 's1_s11_y', 's1_s12_x', 's1_s12_y',
                's2_s11_x', 's2_s11_y', 's2_s12_x', 's2_s12_y',
            ]
        else:
            x_set = [
                's1_s11_abs', 's1_s11_ph', 's1_s12_abs', 's1_s12_ph',
                's2_s11_abs', 's2_s11_ph', 's2_s12_abs', 's2_s12_ph',
            ]
        # --- LEGACY / REFERENCE IMPLEMENTATION ---
        # The following block shows an older compile-in-factory approach.
        # It is intentionally kept for future experimentation.
        # x_set = [
		# 	's1_s11_abs', 's1_s11_ph', 's1_s12_ph',
		# 	's2_s11_abs', 's2_s11_ph', 's2_s12_ph'
		# ]
        # three numeric outputs
        y_set = ["S11m", "S21m", "S21p"]
    elif arch == "LNA":
        topologys = [
            "CS", # '0',
            "Cascode", # '1',
            # Combinations of topologies datasets
            # We don't have self combinations.
            # 'combo_0_1',
        ]
        params = [
			'w', 'freq',
            # input s-parameters (8): abs and phase for s11, s12, s21, s22
			'in_s11_abs', 'in_s11_ph', 'in_s12_abs', 'in_s12_ph',             
            'in_s21_abs', 'in_s21_ph', 'in_s22_abs', 'in_s22_ph',
            # output s-parameters (8): abs and phase for s11, s12, s21, s22
			'out_s11_abs', 'out_s11_ph', 'out_s12_abs', 'out_s12_ph',
            'out_s21_abs', 'out_s21_ph', 'out_s22_abs', 'out_s22_ph'
        ]
        if cartesian:
            x_set = [
                "in_s11_x", "in_s11_y", "in_s12_x", "in_s12_y",
                "out_s11_x", "out_s11_y", "out_s12_x", "out_s12_y",
                "w",
            ]
        else:
            x_set = [
                "in_s11_abs", "in_s11_ph", "in_s12_abs", "in_s12_ph",
                "out_s11_abs", "out_s11_ph", "out_s12_abs", "out_s12_ph",
                "w",
            ]
        # freq
        # in_s11_abs	in_s11_ph	in_s12_abs	in_s12_ph	in_s21_abs	in_s21_ph	in_s22_abs	in_s22_ph
        # out_s11_abs	out_s11_ph	out_s12_abs	out_s12_ph	out_s21_abs	out_s21_ph	out_s22_abs	out_s22_ph, w
        # GA GP	GT	NF	b1f	kf
		# The input parameters has one more parameter than phase shifter, the "w".
        y_set = ["GA", "GP", "GT", "NF", "b1f", "kf"]
    return topologys, x_set, y_set


def topology_policy(model_type: str) -> dict:
    """Decide whether a model should receive topology one-hot features as numeric inputs."""
    policies = {
        "FCN":                  {"use_topology_features": False},
        "CCI":                  {"use_topology_features": False},
        "GNN":                  {"use_topology_features": False},
        "CCIGNN-Sep":           {"use_topology_features": False},
    }
    # default: conservative → no topology features
    return policies.get(model_type, {"use_topology_features": False})



def model_selector(model_type, config, use_topology_features: bool = False, features=None, targets=None):
    """Build the parameter dictionary that drives adapter and model creation."""
    features = list(config.features if features is None else features)
    targets = list(config.targets if targets is None else targets)
    main_arch = config.main_arch
    top_prefix = config.TOPOLOGY_PREFIX
    params = {
        "io": {},
        "arch": {},
        "meta": {},
        "requirements": {}
    }
    params["meta"]["model_type"] = model_type
    params["meta"]["main_arch"] = main_arch
    # Model type based on the main architecture
    if model_type in ("FCN"):
        # FCN (Fully Connected Network)
        params["io"] = {
            # "input_params": ['s1_s11_abs', 's1_s11_ph', 's1_s12_abs', 's1_s12_ph', 's2_s11_abs', 's2_s11_ph', 's2_s12_abs', 's2_s12_ph'],
            "inputs": features,
            # Use the targets defined by the architecture (phase_shifter → ["S11m"], LNA → ["GA"], etc.)
            "outputs": targets,
            "mode": "flat",
        }
        params["arch"] = {
            # Neuron count and activation functions for hidden layers
            "nodes": [128, 256, 128],
            "activation": "relu",
        }
        nodes_phrase = f"N={'-'.join(map(str, params['arch']['nodes']))}"
    elif model_type in ("CCI"):
        # CCI-NN (Circuit Connectivity Inspired Neural Network)
        # Split features into base RF features and optional topology one-hots
        base_features = [f for f in features if not f.startswith(top_prefix)]
        topo_features = [f for f in features if f.startswith(top_prefix)]
        sub_n_nodes = [[128, 128, 128], [128, 128, 128]]
        if main_arch == "phase_shifter":
            label1, label2 = "s1_", "s2_"
            group1 = [f for f in base_features if f.startswith(label1)]
            group2 = [f for f in base_features if f.startswith(label2)]
            if not group1 or not group2:
                raise RuntimeError(f"[CCI-NN][{main_arch}] Missing feature groups for prefixes {label1} or {label2}. base_features={base_features}")
            input_groups = [group1, group2]
            # If we have topology one-hot columns, add them as a separate subnet
            # Add topology as third subnet ONLY for TopCon
            if use_topology_features:
                if topo_features:
                    input_groups.append(topo_features)
                    sub_n_nodes.append([16, 8])
            if config.debug:
                print(f"[{model_type}][{main_arch}] Input groups: {input_groups}")
            params["io"] = {
                "inputs": base_features,
                "input_groups": input_groups,
                "outputs": targets,
                "mode": "grouped",
            }
        elif main_arch == "LNA":
            late_features = [f for f in base_features if f == "w"]
            branch_base_features = [f for f in base_features if f != "w"]
            # Architecture-aware grouping
            # For LNA, group by in_* vs out_*
            label1, label2 = "in_", "out_"
            group1 = [f for f in branch_base_features if f.startswith(label1)]
            group2 = [f for f in branch_base_features if f.startswith(label2)]
            # Fallback: if something is wrong (e.g., prefixes missing), use the old half-split
            if not group1 or not group2:
                raise RuntimeError(f"[CCI-NN][{main_arch}] Missing in_/out_ groups. branch_base_features={branch_base_features}")
            input_groups = [group1, group2]
            # If we have topology one-hot columns, add them as a separate subnet
            if use_topology_features:
                if topo_features:
                    input_groups.append(topo_features)
                    sub_n_nodes.append([16, 8])
            if config.debug:
                print(f"[{model_type}][{main_arch}] input groups: {input_groups}")
                print(f"[{model_type}][{main_arch}] late features: {late_features}")
            params["io"] = {
                "inputs": base_features,
                "input_groups": input_groups,
                "late_features": late_features,
                "outputs": targets,
                "mode": "grouped",
            }
        params["arch"] = {
            # Sub-nets neuron count and activation functions
            "subnets": sub_n_nodes,
            "sub_activation": "relu",
            # Let's call the final block of CCI-NN the "accum" where all the sub-nets' effects would be accumulated.
            # accum layer neuron count and activation functions
            "accum_nodes": [512],
            "accum_activation": "relu",
        }
        sub_nodes_concat = "&".join(f"SN{i+1}={'-'.join(map(str, sublist))}" for i, sublist in enumerate(sub_n_nodes))
        accum_nodes_concat = f"AN={'-'.join(map(str, params['arch']['accum_nodes']))}"
        nodes_phrase = f"{sub_nodes_concat}_{accum_nodes_concat}"
    elif model_type in ("GNN"):
        # Drop topology one-hot from the feature list, but KEEP topology_index
        base_features = [f for f in features if not (f.startswith(top_prefix) and f != config.TOPOLOGY_COL)]
        params["io"] = {
            "inputs": base_features,     # effective dense inputs used to build X (plus topo_index in DF)
            "outputs": targets,
            "mode": "gnn",
            "keeps_topology_index": True,
            "drops_topology_onehot": True,
        }
        params["arch"] = {
            "nodes": [64, 64, 64],
            "activation": "relu",
            "attention": (model_type == "GNN-ATT"),
        }
        params["requirements"] = {
            "graph": True,
        }
        # Naming: suffix "_ATT" only for the attention variant
        suffix = "" if model_type == "GNN" else "_ATT"
        nodes_phrase = f"N={'-'.join(map(str, params['arch']['nodes']))}{suffix}"
    elif model_type == "CCIGNN-Sep":
        if main_arch == "phase_shifter":
            # CCI-GNN: CCI encoder (no topology features) + GNN (uses topology_index structurally)
            # CCI must NOT see topology_* or topology_index
            base_features = [f for f in features if not (f.startswith(top_prefix) or f == config.TOPOLOGY_COL)]
            # phase_shifter (and any other arch): s1_* vs s2_*
            label1, label2 = "s1_", "s2_"    
            group1 = [f for f in base_features if f.startswith(label1)]
            group2 = [f for f in base_features if f.startswith(label2)]
            if not group1 or not group2:
                raise RuntimeError(f"[CCI-GNN] Expected both {label1}* and {label2}* feature groups for CCI branch. base_features={base_features}")
            params["io"] = {
                "inputs": base_features,                  # required by DenseAdapter
                "cci_input_groups": [group1, group2],    # required by CCIGNNAdapter
                "late_features": [],                     # keep schema consistent
                "outputs": targets,
                # Hybrid input order contract (important):
                # adapter returns: [cci1, cci2, X, A]
                "mode": "hybrid_cci+gnn",
                "input_order": ["cci_group_1", "cci_group_2", "X", "A"],
            }
            params["arch"] = {
                # ---- CCI branch ----
                "cci": {
                    "subnets": [[64, 64], [64, 64]],
                    "sub_activation": "relu",
                    "accum_nodes": [128, 64],
                    "accum_activation": "relu",
                },
                # ---- GNN branch ----
                "gnn": {
                    # NOTE: GNN will consume topology_index via adapter
                    "nodes": [64, 64, 64],
                    "activation": "relu",
                }
            }
            params["requirements"] = {
                # Adapter MUST inject these before model creation
                "graph": True,
                "node_shape": True,
            }
        elif main_arch == "LNA":
            # CCI-GNN for LNA:
            # CCI side should follow the same semantic split as CCI-NN (LNA):
            #   - late feature: w
            #   - branch groups: in_* and out_*
            # GNN side still uses topology_index structurally.
            base_features = [f for f in features if not f.startswith(top_prefix)]
            late_features = [f for f in base_features if f == "w"]
            branch_base_features = [f for f in base_features if f != "w"]

            # LNA: group by in_* vs out_*
            label1, label2 = "in_", "out_"
            group1 = [f for f in branch_base_features if f.startswith(label1)]
            group2 = [f for f in branch_base_features if f.startswith(label2)]
            if not group1 or not group2:
                raise RuntimeError(f"[CCI-GNN][LNA] Missing in_/out_ groups. branch_base_features={branch_base_features}")
            if config.debug:
                print(f"[CCI-GNN][LNA] cci_input_groups: {[group1, group2]}")
                print(f"[CCI-GNN][LNA] late_features: {late_features}")
            params["io"] = {
                "inputs": base_features,
                "cci_input_groups": [group1, group2],
                "late_features": late_features,
                "outputs": targets,
                "mode": "hybrid_cci+gnn",
                "input_order": ["cci_group_1", "cci_group_2", "late_features", "X", "A"],
            }
            params["arch"] = {
                "cci": {
                    "subnets": [[64, 64], [64, 64]],
                    "sub_activation": "relu",
                    "accum_nodes": [128, 64],
                    "accum_activation": "relu",
                },
                "gnn": {
                    "nodes": [64, 64, 64],
                    "activation": "relu",
                }
            }
            params["requirements"] = {
                "graph": True,
                "node_shape": True,
            }
        nodes_phrase = "CCIGNN-Sep"
    params["meta"]["has_topology_input"] = any(f.startswith(top_prefix) for f in features)
    return params, nodes_phrase


# Create the requested model from the prepared parameter dictionary.
def create_model(model_type, model_parameters, config, prefix_string):
    """Instantiate the selected model architecture from its parameter dictionary."""
    io = model_parameters.get("io", {})
    if model_type in ("FCN-Top", "FCN", "FCN-SepTopNet"):
        if "inputs" not in io or not io["inputs"]:
            raise RuntimeError(f"[INPUT SHAPE ERROR] {model_type} has no input features defined.")
    if model_type in ("FCN-Top", "FCN"):
        return FC_net(model_parameters, prefix_string=prefix_string)
    elif model_type == "CCI":
        return CCI_net(model_parameters, prefix_string=prefix_string)
    elif model_type == "GNN":
        return GNN_net(model_parameters, prefix_string=prefix_string)
    elif model_type == "CCIGNN-Sep":
        return CCIGNN_Sep_net(model_parameters=model_parameters, prefix_string=prefix_string)
    else:
        raise ValueError(f"Model {model_type} is not defined for creating a model.")


def get_shape(obj):
    """Return a simple nested shape description for lists used in model naming."""
    if isinstance(obj, list):
        if not obj:   # empty list
            return (0,)
        return (len(obj),) + get_shape(obj[0])
    else:
        return ()

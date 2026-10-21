# code/models/cci_gnn.py
"""TARGet"""
from typing import Dict, Any

from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from spektral.layers import GCNConv, GlobalAvgPool


def CCIGNN_Sep_net(model_parameters: Dict[str, Any], prefix_string: str = "ccignn_Sep") -> Model:
    """Build the hybrid model that fuses CCI branch embeddings with a graph encoder."""
    # Unpack and validate the configuration assembled by the factory and adapter.
    io = model_parameters["io"]
    arch = model_parameters["arch"]
    cci_arch = arch["cci"]
    gnn_arch = arch["gnn"]
    cci_input_groups = io["cci_input_groups"]
    late_features = io.get("late_features", [])
    cci_sub_n_nodes = cci_arch["subnets"]
    cci_sub_act = cci_arch["sub_activation"]
    cci_accum_nodes = cci_arch["accum_nodes"]
    cci_accum_act = cci_arch["accum_activation"]
    gnn_nodes_count = gnn_arch["nodes"]
    gnn_activation = gnn_arch["activation"]
    output_params = io["outputs"]
    n_outputs = len(output_params)
    if "N" not in model_parameters or "F" not in model_parameters:
        raise RuntimeError("[CCIGNN_Sep_net] model_parameters must contain 'N' and 'F' (injected by CCIGNNAdapter) before building the model.")
    N: int = model_parameters["N"]
    F: int = model_parameters["F"]
    if len(cci_input_groups) != len(cci_sub_n_nodes):
        raise ValueError(f"[CCIGNN_Sep_net] cci_input_groups (len={len(cci_input_groups)}) and cci_sub_n_nodes (len={len(cci_sub_n_nodes)}) must have the same length.")
    cci_inputs = []
    cci_sub_outputs = []
    for i, (group, sub_nodes) in enumerate(zip(cci_input_groups, cci_sub_n_nodes), start=1):
        in_dim = len(group)
        inp = Input(shape=(in_dim,), name=f"{prefix_string}_cci_in_{i}")
        x = inp
        for j, units in enumerate(sub_nodes, start=1):
            x = Dense(units, activation=cci_sub_act, name=f"{prefix_string}_cci_sub{i}_dense{j}",)(x)
        cci_inputs.append(inp)
        cci_sub_outputs.append(x)
    if len(cci_sub_outputs) > 1:
        accum = Concatenate(name=f"{prefix_string}_cci_concat")(cci_sub_outputs)
    else:
        accum = cci_sub_outputs[0]
    for k, units in enumerate(cci_accum_nodes, start=1):
        accum = Dense(units, activation=cci_accum_act, name=f"{prefix_string}_cci_accum_dense{k}",)(accum)
    cci_embedding = accum
    x_nodes_in = Input(shape=(N, F), name=f"{prefix_string}_X_in")
    a_in = Input(shape=(N, N), name=f"{prefix_string}_A_in")
    h = x_nodes_in
    for idx, units in enumerate(gnn_nodes_count, start=1):
        h = GCNConv(units, activation=gnn_activation, mode="batch", name=f"{prefix_string}_gcn{idx}",)([h, a_in])
    graph_embedding = GlobalAvgPool(name=f"{prefix_string}_readout")(h) # g
    # Fusion head: concat([z, g]) → MLP → outputs
    fusion_inputs = [cci_embedding, graph_embedding]
    if late_features:
        late_in = Input(shape=(len(late_features),), name=f"{prefix_string}_late_in")
        fusion_inputs.append(late_in)
    fused = Concatenate(name=f"{prefix_string}_fusion")(fusion_inputs)
    h = Dense(64, activation="relu", name=f"{prefix_string}_head_dense1")(fused)
    h = Dense(32, activation="relu", name=f"{prefix_string}_head_dense2")(h)
    output = Dense(n_outputs, name=f"{prefix_string}_output")(h)
    # Input ordering must match the tensor list returned by the hybrid adapter.
    model_inputs = list(cci_inputs)
    if late_features:
        model_inputs.append(late_in)
    model_inputs += [x_nodes_in, a_in]
    model = Model(inputs=model_inputs, outputs=output, name=f"{prefix_string}_model")
    return model

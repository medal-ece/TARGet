# code/models/cci_nn.py
"""Circuit Connectivity Inspired neural network."""

from keras.models import Model
from keras.layers import Dense, Input, Concatenate


def CCI_net(model_parameters, prefix_string="CCI"):
    """Build the grouped-input CCI network used for circuit branches."""
    io = model_parameters["io"]
    arch = model_parameters["arch"]
    input_params = io["input_groups"]
    has_topology_branch = len(input_params) > 2
    base_input_params = input_params[:-1] if has_topology_branch else input_params
    output_params = io["outputs"]
    late_features = io.get("late_features", [])
    sub_nets_nodes = arch["subnets"]
    base_sub_nets_nodes = sub_nets_nodes[:-1] if has_topology_branch else sub_nets_nodes
    sub_net_AF = arch["sub_activation"]
    accum_node_count = arch["accum_nodes"]
    accumulator_AF = arch["accum_activation"]
    input_shapes = [(len(sub_inputs),) for sub_inputs in base_input_params]
    inputs = []
    outputs = []
    for i, shape in enumerate(input_shapes):
        subnet_input = Input(shape=shape, name=f"{prefix_string}_subnet_input_{i+1}")
        inputs.append(subnet_input)
        subnet_layer_input = subnet_input if i == 0 else Concatenate(name=f"{prefix_string}_concat{i}")([subnet_input, outputs[i-1]])
        for j, node_count in enumerate(base_sub_nets_nodes[i]):
            subnet_layer_input = Dense(node_count, activation=sub_net_AF, name=f"{prefix_string}_subdense_{i+1}_{j}")(subnet_layer_input)
        layer_output = subnet_layer_input
        outputs.append(layer_output)
    # Merge branch embeddings before the final prediction head.
    if len(outputs) == 1:
        accum_layer = outputs[0]
    else:
        accum_layer = Concatenate(name=f"{prefix_string}_merge")(outputs)
    # Late features such as `w` are fused after the branch encoders.
    if late_features:
        late_input = Input(shape=(len(late_features),), name=f"{prefix_string}_late_input")
        inputs.append(late_input)
        accum_layer = Concatenate(name=f"{prefix_string}_late_merge")([accum_layer, late_input])
    for i, node_count in enumerate(accum_node_count):
        accum_layer = Dense(node_count, activation=accumulator_AF, name=f"{prefix_string}_accum_dense_{i+1}")(accum_layer)
    output_layer = Dense(len(output_params), name=f"{prefix_string}_output_layer")(accum_layer)
    model = Model(inputs=inputs, outputs=output_layer, name=f"{prefix_string}_model")
    return model

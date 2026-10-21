# code/models/fcn.py
"""Fully connected baseline model."""

from keras.models import Sequential
from keras.layers import Dense

def FC_net(model_parameters, prefix_string="fcn"):
    """Build the project's fully connected baseline network."""
    io = model_parameters["io"]
    arch = model_parameters["arch"]
    input_params  = io["inputs"]
    output_params = io["outputs"]
    nodes_count   = arch["nodes"]
    activation    = arch["activation"]
    model = Sequential(name=f"{prefix_string}_model")
    for i, node_count in enumerate(nodes_count):
        if i == 0:
            model.add(Dense(node_count, activation=activation, input_shape=(len(input_params),), name=f"{prefix_string}_input"))
        else:
            model.add(Dense(node_count, activation=activation, name=f"{prefix_string}_hidden{i+1}"))
    model.add(Dense(len(output_params), name=f"{prefix_string}_output"))
    return model

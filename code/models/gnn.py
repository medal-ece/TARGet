# code/models/gnn.py
"""Graph dataset and GNN model definitions."""

from torch_geometric.data import Dataset
import numpy as np
from spektral.data import Dataset, Graph
from spektral.layers import GCNConv, GlobalAvgPool
from keras.models import Model
from keras.layers import Input, Dense


class CircuitDataset(Dataset):
    def __init__(self, data_by_freq, topology_col="topology_index", main_arch="phase_shifter", target_cols=None, **kwargs):
        """Wrap a single-frequency DataFrame and expose it as a graph dataset."""
        self.data_by_freq = data_by_freq
        self.topology_col = topology_col
        self.main_arch = main_arch
        self.target_cols = target_cols or ["S11m"]
        super().__init__(**kwargs)

    def build_node_features(self, row, topo_idx):
        """Build the node feature matrix for one circuit sample."""
        def port_features(prefix):
            f11x = f"{prefix}s11_x"
            f11y = f"{prefix}s11_y"
            f12x = f"{prefix}s12_x"
            f12y = f"{prefix}s12_y"
            for col in (f11x, f11y, f12x, f12y):
                if col not in row.index:
                    raise KeyError(f"[CircuitDataset] Expected column '{col}' in row for main_arch='{self.main_arch}'. Available columns: {list(row.index)}")
            return [row[f11x], row[f11y], row[f12x], row[f12y]]
        
        def vectorize(name, NT, encoding="decimal"):
            idx = NT[name]
            if encoding == "decimal":
                vec = [idx]
            else:
                vec = [0] * len(NT)
                vec[idx] = 1
            return vec
        
        if self.main_arch == "phase_shifter":
            node_count = 4
            # Expect Cartesian features from diagnostics: s1_*, s2_*
            in_prefix = "s1_"
            out_prefix = "s2_"
            NODE_TYPES = {
                # 
                "input": 0,
                "output": 2,
                # S-parameters blocks:
                "first_block": 1,
                "second_block": 1,
            }
            n1 = vectorize("input", NODE_TYPES) + [0, 0, 0, 0]                       # constant descriptor (same as before)
            n2 = vectorize("first_block", NODE_TYPES) + port_features(in_prefix)     # "input" block
            n3 = vectorize("second_block", NODE_TYPES) + port_features(out_prefix)   # "output" block
            n4 = vectorize("output", NODE_TYPES) + [0, 0, 0, 0]                      # constant descriptor
            X = np.array([n1, n2, n3, n4], dtype=float)
        elif self.main_arch == "LNA":
            node_count = 6
            # Expect Cartesian features for in_* and out_*, plus "w" as a late feature
            in_prefix = "in_"
            out_prefix = "out_"
            # Fixed LNA node order:
            # 0: source
            # 1: input block
            # 2: common-source transistor
            # 3: topology-dependent middle node
            #    - Cascode: common-gate transistor
            #    - CS:      wire
            # 4: output block
            NODE_TYPES = {
                "input": 0,
                # S-parameters blocks:
                "first_block": 1,
                "second_block": 1,
                # Transistor type:
                "cs_transistor": 2,
                "cg_transistor": 3,
                # Output block:
                "vcc": 5,
                # Componenet type (for the middle node):
                "wire": 4,
            }
            if row["w"] == 4:
                # W = 4
                w_val = [0.5, 0, 0, 0]
            else:
                # W = 8
                w_val = [1.0, 0, 0, 0]
            n1 = vectorize("input", NODE_TYPES) + [0, 0, 0, 0]                 # source
            n2 = vectorize("first_block", NODE_TYPES) + port_features(in_prefix)  # input block
            n3 = vectorize("cs_transistor", NODE_TYPES) + w_val                 # common-source transistor
            if topo_idx == 0:                    # CS
                n4 = vectorize("wire", NODE_TYPES) + [0, 0, 0, 0]             # wire
            elif topo_idx == 1:                  # Cascode
                n4 = vectorize("cg_transistor", NODE_TYPES) + w_val             # common-gate transistor
            else:
                raise ValueError(f"[CircuitDataset] Unknown LNA topology index: {topo_idx}")
            n5 = vectorize("second_block", NODE_TYPES) + port_features(out_prefix) # output block
            n6 = vectorize("vcc", NODE_TYPES) + [0, 0, 0, 0]                 # VCC
            X = np.array([n1, n2, n3, n4, n5, n6], dtype=float)
        else:
            raise ValueError(f"[CircuitDataset] Unknown main_arch='{self.main_arch}'")
        node_id = np.eye(node_count, dtype=float)
        X = np.concatenate([X, node_id], axis=1)
        return X

    def build_adjacency(self, topo_idx):
        """Return the fixed adjacency matrix for the requested topology."""
        if self.main_arch == "phase_shifter":
            if topo_idx == 0:   # sw1_0_sw2_0
                A = np.array([
                    [1,1,0,0],
                    [1,1,1,0],
                    [0,1,1,1],
                    [0,0,1,1]
                ], dtype=float)
            elif topo_idx == 1:  # sw1_0_sw2_1
                A = np.array([
                    [1,1,0,0],
                    [1,1,0,1],
                    [0,0,1,0],
                    [0,1,0,1]
                ], dtype=float)
            elif topo_idx == 2:   # sw1_1_sw2_0
                A = np.array([
                    [1,0,1,0],
                    [0,1,0,0],
                    [1,0,1,1],
                    [0,0,1,1]
                ], dtype=float)
            elif topo_idx == 3:  # sw1_1_sw2_1
                A = np.array([
                    [1,0,0,1],
                    [0,1,0,0],
                    [0,0,1,0],
                    [1,0,0,1]
                ], dtype=float)
            else:
                raise ValueError(f"[CircuitDataset] Unknown phase_shifter topology index: {topo_idx}")
        elif self.main_arch == "LNA":
            # Fixed LNA node order:
            # 0: source
            # 1: input block
            # 2: common-source transistor
            # 3: topology-dependent middle node
            #    - CS:      wire
            #    - Cascode: common-gate transistor
            # 4: output block
            # 5: VCC
            if topo_idx == 0:  # CS
                # source - input - CS transistor - wire - output
                # VCC connected to transistor side
                A = np.array([
                    [1, 1, 0, 0, 0, 0],  # source
                    [1, 1, 1, 0, 0, 0],  # input block
                    [0, 1, 1, 1, 1, 0],  # CS transistor
                    [0, 0, 1, 1, 1, 0],  # wire
                    [0, 0, 1, 1, 1, 1],  # output block
                    [0, 0, 0, 0, 1, 1],  # VCC
                ], dtype=float)
            elif topo_idx == 1:  # Cascode
                # source - input - CS transistor - CG transistor - output
                # VCC connected to common-gate side
                A = np.array([
                    [1, 1, 0, 0, 0, 0],  # source
                    [1, 1, 1, 0, 0, 0],  # input block
                    [0, 1, 1, 1, 0, 0],  # CS transistor
                    [0, 0, 1, 1, 1, 0],  # CG transistor
                    [0, 0, 0, 1, 1, 1],  # output block
                    [0, 0, 0, 0, 1, 1],  # VCC
                ], dtype=float)
            else:
                raise ValueError(f"[CircuitDataset] Unknown LNA topology index: {topo_idx}")
        else:
            raise ValueError(f"[CircuitDataset] Unknown main_arch='{self.main_arch}'")
        return A

    def build_target(self, row):
        """Build the target vector for one row using the configured target columns."""
        values = []
        for col in self.target_cols:
            if col not in row.index:
                raise KeyError(f"[CircuitDataset] Expected target column '{col}' in row. Available columns: {list(row.index)}")
            values.append(float(row[col]))
        y = np.array([values], dtype=float)
        return y

    def read(self):
        df = self.data_by_freq
        graphs = []
        for _, row in df.iterrows():
            topo_idx = int(row[self.topology_col])
            X = self.build_node_features(row, topo_idx)
            A = self.build_adjacency(topo_idx)
            if X.shape[0] != A.shape[0] or A.shape[0] != A.shape[1]:
                raise ValueError(f"[CircuitDataset] Inconsistent graph shapes for main_arch='{self.main_arch}', topo_idx={topo_idx}: X.shape={X.shape}, A.shape={A.shape}")
            y = self.build_target(row)
            graphs.append(Graph(x=X, a=A, y=y))
        return graphs



# Build the graph model used for training.
def GNN_net(model_parameters, prefix_string="gnn"):
    """Build the graph neural network used by the project."""
    io = model_parameters["io"]
    arch = model_parameters["arch"]

    output_params = io["outputs"]
    nodes_count = arch["nodes"]
    activation = arch["activation"]
    n_outputs = len(output_params)

    # These dimensions are injected by the adapter once graph tensors are built.
    N = model_parameters.get("N", None)
    F = model_parameters.get("F", None)
    if N is None or F is None:
        raise RuntimeError("[GNN_net] model_parameters['N'] and ['F'] must be set by the adapter.")

    x_in = Input(shape=(N, F), name=f"{prefix_string}_X_in")
    a_in = Input(shape=(N, N), name=f"{prefix_string}_A_in")
    h = GCNConv(nodes_count[0], activation=activation, mode='batch', name=f"{prefix_string}_gcn1")([x_in, a_in])
    for i, c in enumerate(nodes_count[1:], start=2):
        h = GCNConv(c, activation=activation, mode='batch', name=f"{prefix_string}_gcn{i}")([h, a_in])
    hg = GlobalAvgPool(name=f"{prefix_string}_readout")(h)
    hg = Dense(64, activation="relu", name=f"{prefix_string}_dense1")(hg)
    hg = Dense(32, activation="relu", name=f"{prefix_string}_dense2")(hg)
    output = Dense(n_outputs, name=f"{prefix_string}_output")(hg)
    model = Model(inputs=[x_in, a_in], outputs=output, name=f"{prefix_string}_model")
    return model

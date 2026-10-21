# data/adapters.py

# code/data/adapters.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from data.loaders import data_preparation
from models.gnn import CircuitDataset


class AdapterMode:
    """Supported adapter behaviors for grouped-input models."""

    MIRROR = "mirror"
    SEMANTIC = "semantic"


@dataclass
class AdapterOutput:
    """
    Standardized output of all adapters.
    x_train, x_test:
        - FCN: np.ndarray, shape (N, F)
        - CCI-NN: List[np.ndarray], each shape (N, Fi)
        - GNN: [X, A], each shape (N, nodes, feat) / (N, nodes, nodes)
    y_train, y_test: np.ndarray, shape (N, n_targets)
    df_train_scaled, df_test_scaled: Scaled DataFrames (useful for diagnostics / GNN graph building)
    xscaler, yscaler: Fitted scalers (for inverse-transform / saving).
    """
    x_train: Any
    y_train: np.ndarray
    x_test: Any
    y_test: np.ndarray
    df_train_scaled: pd.DataFrame
    df_test_scaled: pd.DataFrame
    xscaler: Any
    yscaler: Any


class BaseDataAdapter:
    """Base interface for turning cleaned DataFrames into model-ready tensors."""
    def prepare(self, df_train: pd.DataFrame, df_test: pd.DataFrame, config, model_parameters: Dict[str, Any], scaler) -> AdapterOutput:
        raise NotImplementedError

class DenseAdapter(BaseDataAdapter):
    """Adapter for dense models such as the FCN baseline."""
    def prepare(self, df_train: pd.DataFrame, df_test: pd.DataFrame, config, model_parameters: Dict[str, Any], scaler) -> AdapterOutput:
        io = model_parameters.get("io", {})
        inputs = io.get("inputs")
        if not inputs:
            raise RuntimeError("[DenseAdapter] params['io']['inputs'] is required")
        outputs = io.get("outputs")
        if not outputs:
            raise RuntimeError("[DenseAdapter] params['io']['outputs'] is required")
        # Dense models work on flat numeric features only.
        if config.TOPOLOGY_COL in inputs:
            raise RuntimeError(f"[DenseAdapter] '{config.TOPOLOGY_COL}' must not be in dense model inputs")
        data = data_preparation(inputs=inputs, outputs=outputs, df_train=df_train, df_test=df_test, scaler=scaler)
        return AdapterOutput(
            x_train=data["x_train"],
            y_train=data["y_train"],
            x_test=data["x_test"],
            y_test=data["y_test"],
            df_train_scaled=data["df_train_scaled"],
            df_test_scaled=data["df_test_scaled"],
            xscaler=data["xscaler"],
            yscaler=data["yscaler"],
        )


class CCINNAdapter(BaseDataAdapter):
    """Adapter for CCI-NN in either legacy mirror mode or semantic grouping mode."""
    def __init__(self, mode=AdapterMode.MIRROR):
        self.mode = mode
    
    def prepare(self, df_train: pd.DataFrame, df_test: pd.DataFrame, config, model_parameters: Dict[str, Any], scaler) -> AdapterOutput:
        # Both modes share the same scaling path.
        dense = DenseAdapter().prepare(df_train, df_test, config, model_parameters, scaler)
        if self.mode == AdapterMode.MIRROR:
            # Preserve the original positional two-branch split.
            x_tr = dense.x_train  # np.ndarray
            x_te = dense.x_test
            io = model_parameters.get("io", {})
            effective_inputs = io.get("inputs")
            if not effective_inputs:
                raise RuntimeError("[CCINNAdapter][MIRROR] params['io']['inputs'] is required")
            if len(effective_inputs) % 2 != 0:
                raise RuntimeError(f"[CCINNAdapter][MIRROR] Expected even number of effective inputs, got {len(effective_inputs)}")
            half_params = int(len(effective_inputs) / 2)
            input1_tr = x_tr[:, :half_params]
            input2_tr = x_tr[:, half_params:]
            x_train = [input1_tr, input2_tr]
            input1_te = x_te[:, :half_params]
            input2_te = x_te[:, half_params:]
            x_test = [input1_te, input2_te]
            return AdapterOutput(
                x_train=x_train,
                y_train=dense.y_train,
                x_test=x_test,
                y_test=dense.y_test,
                df_train_scaled=dense.df_train_scaled,
                df_test_scaled=dense.df_test_scaled,
                xscaler=dense.xscaler,
                yscaler=dense.yscaler,
            )
        elif self.mode == AdapterMode.SEMANTIC:
            # Group inputs by named feature sets instead of column position.
            io = model_parameters.get("io", {})
            input_groups: List[List[str]] = io.get("input_groups")
            topology_features: List[str] = io.get("topology_features", [])
            late_features: List[str] = io.get("late_features", [])
            if not input_groups:
                raise RuntimeError("[CCINNAdapter][SEMANTIC] params['io']['input_groups'] is required")
            # topology_index is structural metadata, not a semantic branch input.
            for group in input_groups:
                if config.TOPOLOGY_COL in group:
                    raise RuntimeError("[CCINNAdapter][SEMANTIC] topology_index must not be part of input feature groups")

            df_tr_s = dense.df_train_scaled
            df_te_s = dense.df_test_scaled

            # Validate every requested feature before building tensors.
            missing_tr = []
            missing_te = []
            for group in input_groups:
                for feat in group:
                    if feat not in df_tr_s.columns:
                        missing_tr.append(feat)
                    if feat not in df_te_s.columns:
                        missing_te.append(feat)
            for feat in topology_features:
                if feat not in df_tr_s.columns:
                    missing_tr.append(feat)
                if feat not in df_te_s.columns:
                    missing_te.append(feat)
            for feat in late_features:
                if feat not in df_tr_s.columns:
                    missing_tr.append(feat)
                if feat not in df_te_s.columns:
                    missing_te.append(feat)
            if missing_tr or missing_te:
                raise RuntimeError(f"[CCINNAdapter][SEMANTIC] Missing features in scaled DataFrames.\nTrain missing: {sorted(set(missing_tr))}\nTest missing:  {sorted(set(missing_te))}")
            # Keep tensor order aligned with the model definition in the factory.
            x_train = [df_tr_s[group].to_numpy() for group in input_groups]
            x_test  = [df_te_s[group].to_numpy() for group in input_groups]
            if topology_features:
                x_train.append(df_tr_s[topology_features].to_numpy())
                x_test.append(df_te_s[topology_features].to_numpy())
            if late_features:
                x_train.append(df_tr_s[late_features].to_numpy())
                x_test.append(df_te_s[late_features].to_numpy())
            return AdapterOutput(
                x_train=x_train,
                y_train=dense.y_train,
                x_test=x_test,
                y_test=dense.y_test,
                df_train_scaled=df_tr_s,
                df_test_scaled=df_te_s,
                xscaler=dense.xscaler,
                yscaler=dense.yscaler,
            )
        else:
            raise RuntimeError(f"[CCINNAdapter] Unknown mode: {self.mode}")


class GNNAdapter(BaseDataAdapter):
    """Adapter that converts scaled rows into graph tensors for the GNN model."""
    def prepare(self, df_train: pd.DataFrame, df_test: pd.DataFrame, config, model_parameters: Dict[str, Any], scaler) -> AdapterOutput:
        # Reuse the dense scaling path so targets stay consistent across models.
        dense = DenseAdapter().prepare(df_train, df_test, config, model_parameters, scaler)
        df_tr_s = dense.df_train_scaled
        df_te_s = dense.df_test_scaled
        # The graph builder needs topology_index to choose node wiring.
        if config.TOPOLOGY_COL not in df_tr_s.columns:
            raise RuntimeError(f"[GNNAdapter] Required topology column '{config.TOPOLOGY_COL}' missing from train DataFrame.")
        if config.TOPOLOGY_COL not in df_te_s.columns:
            raise RuntimeError(f"[GNNAdapter] Required topology column '{config.TOPOLOGY_COL}' missing from test DataFrame.")
        # topology one-hots are redundant once topology_index drives graph construction.
        topo_oh_cols = [c for c in df_tr_s.columns if c.startswith(config.TOPOLOGY_PREFIX) and c != config.TOPOLOGY_COL]
        if topo_oh_cols:
            df_tr_s = df_tr_s.drop(columns=topo_oh_cols)
            df_te_s = df_te_s.drop(columns=topo_oh_cols)

        # Build one graph per row, then batch them into arrays.
        train_ds = CircuitDataset(df_tr_s, topology_col=config.TOPOLOGY_COL, main_arch=config.main_arch, target_cols=config.targets)
        test_ds  = CircuitDataset(df_te_s, topology_col=config.TOPOLOGY_COL, main_arch=config.main_arch, target_cols=config.targets)
        graphs_tr = train_ds.read()
        graphs_te = test_ds.read()
        X_train = np.stack([g.x for g in graphs_tr])
        A_train = np.stack([g.a for g in graphs_tr])
        y_train = np.stack([g.y for g in graphs_tr]).reshape(-1, len(config.targets))
        X_test  = np.stack([g.x for g in graphs_te])
        A_test  = np.stack([g.a for g in graphs_te])
        y_test  = np.stack([g.y for g in graphs_te]).reshape(-1, len(config.targets))
        # Store graph shape for the model factory.
        N = X_train.shape[1]  # number of nodes
        F = X_train.shape[2]  # node feature dimension
        model_parameters["N"] = N
        model_parameters["F"] = F
        x_train = [X_train, A_train]
        x_test  = [X_test, A_test]
        return AdapterOutput(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            df_train_scaled=df_tr_s,
            df_test_scaled=df_te_s,
            xscaler=dense.xscaler,
            yscaler=dense.yscaler,
        )


class CCIGNNAdapter(BaseDataAdapter):
    """
    Adapter for the hybrid CCI-GNN model.

    Responsibilities:
      - Reuse DenseAdapter scaling (same scaler, same y as other models).
      - Build CCI inputs from semantic groups in params["io"]["cci_input_groups"].
      - Build GNN graph tensors (X, A) from CircuitDataset.
      - Set model_parameters["N"] and ["F"] for the GNN part.

    Output shapes:
      x_train = [cci_in_1, cci_in_2, X_train, A_train]
      x_test  = [cci_in_1_te, cci_in_2_te, X_test, A_test]
    """
    def prepare(self, df_train: pd.DataFrame, df_test: pd.DataFrame, config, model_parameters: Dict[str, Any], scaler) -> AdapterOutput:
        # 1) Reuse existing scaling – keeps behavior consistent with FCN/CCI/GNN
        dense = DenseAdapter().prepare(df_train, df_test, config, model_parameters, scaler)
        df_tr_s = dense.df_train_scaled
        df_te_s = dense.df_test_scaled
        # 2) Build CCI inputs from semantic groups
        io = model_parameters.get("io", {})
        input_groups: List[List[str]] = io.get("cci_input_groups")
        late_features: List[str] = io.get("late_features", [])
        if not input_groups:
            raise RuntimeError("[CCIGNNAdapter] params['io']['cci_input_groups'] must be a non-empty list of feature-name groups.")
        
        # topology_index belongs to the graph side, not the CCI feature groups.
        for group in input_groups:
            if config.TOPOLOGY_COL in group:
                raise RuntimeError("[CCIGNNAdapter] topology_index must not be part of CCI input feature groups")
        # Make sure every requested feature is present before building tensors.
        missing_tr = []
        missing_te = []
        for group in input_groups:
            for feat in group:
                if feat not in df_tr_s.columns:
                    missing_tr.append(feat)
                if feat not in df_te_s.columns:
                    missing_te.append(feat)

        for feat in late_features:
            if feat not in df_tr_s.columns:
                missing_tr.append(feat)
            if feat not in df_te_s.columns:
                missing_te.append(feat)

        if missing_tr or missing_te:
            raise RuntimeError(f"[CCIGNNAdapter] Missing features in scaled DataFrames.\nTrain missing: {sorted(set(missing_tr))}\nTest missing:  {sorted(set(missing_te))}")

        # Build CCI inputs (first two inputs to the hybrid model)
        cci_x_train = [df_tr_s[group].to_numpy() for group in input_groups]
        cci_x_test  = [df_te_s[group].to_numpy() for group in input_groups]

        if late_features:
            cci_x_train.append(df_tr_s[late_features].to_numpy())
            cci_x_test.append(df_te_s[late_features].to_numpy())

        # 3) Build GNN inputs (X, A) from CircuitDataset
        # topology_index is required here because it defines the graph structure.
        if config.TOPOLOGY_COL not in df_tr_s.columns:
            raise RuntimeError(f"[CCIGNNAdapter] Required topology column '{config.TOPOLOGY_COL}' missing from train DataFrame.")
        if config.TOPOLOGY_COL not in df_te_s.columns:
            raise RuntimeError(f"[CCIGNNAdapter] Required topology column '{config.TOPOLOGY_COL}' missing from test DataFrame.")

        # Drop topology one-hot columns before graph construction.
        topo_oh_cols = [c for c in df_tr_s.columns if c.startswith(config.TOPOLOGY_PREFIX) and c != config.TOPOLOGY_COL]

        df_tr_gnn = df_tr_s
        df_te_gnn = df_te_s
        if topo_oh_cols:
            df_tr_gnn = df_tr_gnn.drop(columns=topo_oh_cols)
            df_te_gnn = df_te_gnn.drop(columns=topo_oh_cols)

        # Build graph objects per row
        train_ds = CircuitDataset(df_tr_gnn, topology_col=config.TOPOLOGY_COL, main_arch=config.main_arch, target_cols=config.targets)
        test_ds = CircuitDataset(df_te_gnn, topology_col=config.TOPOLOGY_COL, main_arch=config.main_arch, target_cols=config.targets)
        graphs_tr = train_ds.read()
        graphs_te = test_ds.read()

        # Stack into batched numpy arrays
        X_train = np.stack([g.x for g in graphs_tr])
        A_train = np.stack([g.a for g in graphs_tr])
        X_test  = np.stack([g.x for g in graphs_te])
        A_test  = np.stack([g.a for g in graphs_te])

        # Infer graph shape for the GNN branch
        N = X_train.shape[1]  # number of nodes
        F = X_train.shape[2]  # node feature dimension
        model_parameters["N"] = N
        model_parameters["F"] = F

        # Final hybrid inputs:
        #   [CCI sub-inputs..., X, A]
        x_train = cci_x_train + [X_train, A_train]
        x_test  = cci_x_test  + [X_test, A_test]

        return AdapterOutput(
            x_train=x_train,
            y_train=dense.y_train,      # Same y as other models (no change)
            x_test=x_test,
            y_test=dense.y_test,
            df_train_scaled=df_tr_s,    # full scaled DF (with topo one-hots if present)
            df_test_scaled=df_te_s,
            xscaler=dense.xscaler,
            yscaler=dense.yscaler,
        )

def get_data_adapter(model_type: str, config, use_topology_features: bool = False) -> BaseDataAdapter:
    """
    Factory method to pick the appropriate adapter for a given model_type.
    """
    if model_type in ("FCN"):
        return DenseAdapter()
    if model_type in ("CCI"):
        mode_name = config.ADAPTER_MODES.get(model_type, "semantic")
        if mode_name == "mirror":
            return CCINNAdapter(mode=AdapterMode.MIRROR)
        if mode_name == "semantic":
            return CCINNAdapter(mode=AdapterMode.SEMANTIC)
        raise ValueError(f"[DataAdapter] Unsupported CCI-NN adapter mode '{mode_name}' for {model_type}")
    if model_type in ("GNN"):
        return GNNAdapter()
    if model_type in ["CCIGNN-Sep"]:
        # Hybrid adapter: CCI semantic inputs + GNN graph inputs
        return CCIGNNAdapter()
    raise ValueError(f"[DataAdapter] Unknown model_type: {model_type}")

"""
。

：
- checkpoint  trainer.py ， mol_encoder / sol_encoder / fusion_encoder / mlp ；
-  data_loading.py  FluorSolventDataset ；
-  trainer.py  build_x_vec(batch, system)， system.mlp；
-  cfg.TARGET_NAMES / data_loading.TARGET_COLS ， [abs, emi, plqy, em]；
-  {"plqy", "emi", "em", "abs"} 。

：
1.  model_interface.py。
2. （data_loading.py、model_factory.py、trainer.py、MLP.py、define.py、model.py ）
    Python import 。
   ： MAIN_MODEL_ROOT ；
   ，。
3.  checkpoint  target_mean / target_std， checkpoint cfg  TRAIN_CSV 。
    data/splits/deployment/deployment.csv  cv_fold 。
"""

from __future__ import annotations

import os
import random
import re
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

from config import PROPERTIES

warnings.simplefilter("ignore")


def _log(message: str) -> None:
    print(message, flush=True)






_ENV_ROOT_KEYS = (
    "MAIN_MODEL_ROOT",
    "MODEL_CODE_ROOT",
    "FLUORESCENCE_MODEL_ROOT",
)

_MAIN_IMPORTS_READY = False


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        try:
            p = p.expanduser().resolve()
        except Exception:
            p = p.expanduser().absolute()
        key = str(p)
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def _candidate_roots(checkpoint_path: Optional[str] = None) -> list[Path]:
    roots: list[Path] = []

    for key in _ENV_ROOT_KEYS:
        val = os.environ.get(key)
        if val:
            roots.append(Path(val))

    
    roots.append(Path.cwd())

    
    roots.append(Path(__file__).resolve().parent)

    
    
    
    if checkpoint_path:
        ckpt = Path(checkpoint_path).expanduser()
        if not ckpt.is_absolute():
            ckpt = Path.cwd() / ckpt
        for p in [ckpt.parent, *ckpt.parents]:
            roots.append(p)

    return _unique_paths([p for p in roots if str(p).strip()])


def _insert_candidate_roots(checkpoint_path: Optional[str] = None) -> list[Path]:
    roots = _candidate_roots(checkpoint_path)
    
    
    
    for root in reversed(roots):
        s = str(root)
        sys.path[:] = [existing for existing in sys.path if existing != s]
        sys.path.insert(0, s)
    return roots


def _import_main_modules(checkpoint_path: Optional[str] = None):
    """，。"""
    global _MAIN_IMPORTS_READY
    roots = _insert_candidate_roots(checkpoint_path)

    try:
        import data_loading as dl
        import model_factory as mf
        import trainer as tr
        from torch_geometric.loader import DataLoader as PyGDataLoader
    except Exception as e:
        msg = [
            "。 MAIN_MODEL_ROOT ，",
            "：data_loading.py、model_factory.py、trainer.py、MLP.py、define.py、model.py。",
            "",
            "：",
        ]
        msg.extend([f"  - {p}" for p in roots])
        msg.append("")
        msg.append(f"：{type(e).__name__}: {e}")
        raise ImportError("\n".join(msg)) from e

    _MAIN_IMPORTS_READY = True
    return dl, mf, tr, PyGDataLoader






_PATCHED_MORE_ONLY = False
_ORIGINAL_BUILD_MOL_ENCODER = None


class _GraphTransformerOnlineMOREAdapterOnlyEncoder(nn.Module):
    """
     run_model_training_workflow.py  adapter_only ：
    - mol_vec  online MORE graph vector  LayerNorm adapter ；
    -  Graph Transformer  atom tokens， token-level fusion self-attention 。
    """

    def __init__(self, graph_encoder, more_encoder, more_adapter_cls):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.more_encoder = more_encoder
        self.more_adapter = more_adapter_cls(in_dim=int(more_encoder.graph_dim))
        self.graph_dim = int(self.more_adapter.graph_dim)

    def forward(self, data, return_node_embeddings: bool = False):
        graph_vec, node_emb = self.graph_encoder(data=data, return_node_embeddings=True)
        more_vec = self.more_encoder(data)

        if more_vec.device != graph_vec.device or more_vec.dtype != graph_vec.dtype:
            more_vec = more_vec.to(device=graph_vec.device, dtype=graph_vec.dtype, non_blocking=True)

        adapter_vec = self.more_adapter(more_vec)
        if return_node_embeddings:
            return adapter_vec, node_emb
        return adapter_vec


def _needs_more_only_patch(solute_encoder_type: object) -> bool:
    key = str(solute_encoder_type).strip().lower()
    aliases = {
        "graph_transformer_more_online_ln_adapter_only",
        "graph_transformer_online_more_ln_adapter_only",
        "online_more_ln_adapter_only",
        "more_online_ln_adapter_only",
        "graph_transformer_more_online_ln_adapter_no_gt",
        "graph_transformer_online_more_ln_adapter_no_gt",
        "online_more_ln_adapter_no_gt",
        "more_online_ln_adapter_no_gt",
    }
    return key in aliases or "adapter_only" in key or "no_gt" in key


def _patch_more_only_encoder_if_needed(mf, solute_encoder_type: object) -> None:
    """
     HPO noGT / adapter_only  run_model_training_workflow.py  monkey patch 。
     checkpoint  invoke ， patch。
    """
    global _PATCHED_MORE_ONLY, _ORIGINAL_BUILD_MOL_ENCODER

    if not _needs_more_only_patch(solute_encoder_type):
        return

    if _PATCHED_MORE_ONLY:
        return

    if not hasattr(mf, "_build_mol_encoder"):
        raise AttributeError("model_factory.py  _build_mol_encoder， adapter_only 。")
    if not hasattr(mf, "_build_graph_transformer_encoder"):
        raise AttributeError("model_factory.py  _build_graph_transformer_encoder， adapter_only 。")
    if not hasattr(mf, "OnlineMOREGraphEncoder"):
        raise AttributeError("model_factory.py  OnlineMOREGraphEncoder， adapter_only 。")
    if not hasattr(mf, "MORELayerNormAdapter"):
        raise AttributeError("model_factory.py  MORELayerNormAdapter， adapter_only 。")

    _ORIGINAL_BUILD_MOL_ENCODER = mf._build_mol_encoder

    def _build_mol_encoder_more_only(sample_batch, cfg):
        raw_encoder_type = getattr(cfg, "SOLUTE_ENCODER_TYPE", "graph_transformer")
        if not _needs_more_only_patch(raw_encoder_type):
            return _ORIGINAL_BUILD_MOL_ENCODER(sample_batch, cfg)

        graph_encoder, _ = mf._build_graph_transformer_encoder(sample_batch, cfg)
        more_weight_path = getattr(cfg, "MORE_WEIGHT_PATH", None)
        allow_missing_more = bool(getattr(cfg, "ALLOW_MISSING_MORE_WEIGHT_FOR_CHECKPOINT_LOAD", False))
        if (more_weight_path is None or str(more_weight_path).strip() == "") and not allow_missing_more:
            raise ValueError(
                "adapter_only / noGT  MORE_WEIGHT_PATH， 'MORE.pth'。"
            )

        more_encoder = mf.OnlineMOREGraphEncoder(
            weight_path=more_weight_path,
            num_layer=int(getattr(cfg, "MORE_NUM_LAYER", 5)),
            emb_dim=int(getattr(cfg, "MORE_EMB_DIM", 300)),
            jk=str(getattr(cfg, "MORE_JK", "last")),
            dropout_ratio=float(getattr(cfg, "MORE_DROPOUT_RATIO", 0.0)),
            gnn_type=str(getattr(cfg, "MORE_GNN_TYPE", "gin")),
            pooling=str(getattr(cfg, "MORE_POOLING", "mean")),
            frozen=bool(getattr(cfg, "MORE_FROZEN", True)),
            unfrozen_layers=getattr(cfg, "MORE_UNFROZEN_LAYERS", None),
        ).to(cfg.DEVICE)

        mol_encoder = _GraphTransformerOnlineMOREAdapterOnlyEncoder(
            graph_encoder=graph_encoder,
            more_encoder=more_encoder,
            more_adapter_cls=mf.MORELayerNormAdapter,
        ).to(cfg.DEVICE)

        return mol_encoder, int(mol_encoder.graph_dim)

    mf._build_mol_encoder = _build_mol_encoder_more_only
    _PATCHED_MORE_ONLY = True







def _torch_load(path: str | Path, map_location):
    """ PyTorch  torch.load。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _to_numpy_1d(x, dtype=np.float32) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=dtype).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _cfg_value_missing(cfg: dict, key: str) -> bool:
    if key not in cfg:
        return True
    val = cfg.get(key)
    if val is None:
        return True
    if isinstance(val, str) and val.strip().lower() in {"", "none", "null", "nan"}:
        return True
    return False


def _state_dict_keys(state_dict: object) -> list[str]:
    if not isinstance(state_dict, dict):
        return []
    return [str(k) for k in state_dict.keys()]


def _first_tensor_shape(state_dict: object, predicate) -> Optional[Tuple[int, ...]]:
    if not isinstance(state_dict, dict):
        return None
    for key, value in state_dict.items():
        if predicate(str(key)) and torch.is_tensor(value):
            return tuple(int(x) for x in value.shape)
    return None


def _infer_more_hparams_from_state(mol_state: object, cfg_dict: dict) -> None:
    """ mol_encoder.state_dict  MORE ， checkpoint。"""
    if not isinstance(mol_state, dict):
        return

    emb_shape = _first_tensor_shape(
        mol_state,
        lambda k: "more_encoder" in k and k.endswith("x_embedding1.weight"),
    )
    if emb_shape is not None and len(emb_shape) >= 2:
        cfg_dict["MORE_EMB_DIM"] = int(emb_shape[1])

    layer_indices = []
    for key in _state_dict_keys(mol_state):
        m = re.search(r"more_encoder(?:\.more_encoder)?\.gnns\.(\d+)\.", key)
        if m:
            layer_indices.append(int(m.group(1)))
    if layer_indices:
        cfg_dict["MORE_NUM_LAYER"] = max(layer_indices) + 1

    adapter_shape = _first_tensor_shape(
        mol_state,
        lambda k: "more_adapter" in k and k.endswith("norm.weight"),
    )
    if adapter_shape is not None and len(adapter_shape) == 1:
        adapter_dim = int(adapter_shape[0])
        emb_dim = int(cfg_dict.get("MORE_EMB_DIM", 300))
        num_layer = int(cfg_dict.get("MORE_NUM_LAYER", 5))
        if adapter_dim == int(num_layer + 1) * emb_dim:
            cfg_dict["MORE_JK"] = "concat"
        elif adapter_dim == emb_dim:
            cfg_dict["MORE_JK"] = "last"


def _more_graph_dim_from_cfg(cfg_dict: dict) -> int:
    emb_dim = int(cfg_dict.get("MORE_EMB_DIM", 300))
    num_layer = int(cfg_dict.get("MORE_NUM_LAYER", 5))
    jk = str(cfg_dict.get("MORE_JK", "last")).strip().lower()
    if jk == "concat":
        return int(num_layer + 1) * emb_dim
    return emb_dim


def _infer_mol_graph_dim_from_checkpoint_cfg(cfg_dict: dict) -> Optional[int]:
    try:
        mlp_input_dim = int(cfg_dict.get("MLP_INPUT_DIM"))
    except Exception:
        return None

    try:
        in_dim_sol = int(cfg_dict.get("in_dim_sol", cfg_dict.get("solvent_feat_dim", 0)))
    except Exception:
        in_dim_sol = 0

    try:
        fusion_dim = int(cfg_dict.get("fusion_output_dim", 0))
    except Exception:
        fusion_dim = 0

    mol_dim = mlp_input_dim - in_dim_sol - fusion_dim
    return int(mol_dim) if mol_dim > 0 else None


def _infer_readout_cfg_from_state(mol_state: object, cfg_dict: dict) -> None:
    """ checkpoint  GRAPH_TRANSFORMER_READOUT_*，。"""
    if not isinstance(mol_state, dict):
        return
    keys = _state_dict_keys(mol_state)
    if not any("readout_encoder" in k for k in keys):
        return

    cfg_dict["GRAPH_TRANSFORMER_READOUT"] = "self_attention"

    layer_indices = []
    for key in keys:
        m = re.search(r"readout_encoder\.layers\.(\d+)\.", key)
        if m:
            layer_indices.append(int(m.group(1)))
    if layer_indices:
        cfg_dict["GRAPH_TRANSFORMER_READOUT_LAYERS"] = max(layer_indices) + 1

    ffn_shape = _first_tensor_shape(
        mol_state,
        lambda k: "readout_encoder.layers.0.linear1.weight" in k,
    )
    if ffn_shape is not None and len(ffn_shape) >= 1:
        cfg_dict["GRAPH_TRANSFORMER_READOUT_FFN_DIM"] = int(ffn_shape[0])


def _checkpoint_has_more_state(ckpt: dict) -> bool:
    mol_state = ckpt.get("mol_encoder")
    if not isinstance(mol_state, dict):
        return False
    keys = _state_dict_keys(mol_state)
    return any("more_encoder" in k for k in keys) or any("more_adapter" in k for k in keys)


def _infer_architecture_from_checkpoint_state(ckpt: dict, cfg_dict: dict, raw_cfg: dict) -> None:
    """
     trainer  cfg  checkpoint ：
    -  mol_encoder  MORE / noGT / self-readout；
    -  MLP_INPUT_DIM  adapter_only  gt_concat ；
    -  checkpoint cfg 。
    """
    mol_state = ckpt.get("mol_encoder")
    if not isinstance(mol_state, dict):
        return

    _infer_more_hparams_from_state(mol_state, cfg_dict)
    _infer_readout_cfg_from_state(mol_state, cfg_dict)

    keys = _state_dict_keys(mol_state)
    has_more = any("more_encoder" in k for k in keys) or any("more_adapter" in k for k in keys)
    if not has_more:
        return

    
    cfg_dict["ALLOW_MISSING_MORE_WEIGHT_FOR_CHECKPOINT_LOAD"] = True

    if _cfg_value_missing(raw_cfg, "SOLUTE_ENCODER_TYPE"):
        more_dim = _more_graph_dim_from_cfg(cfg_dict)
        gt_dim = int(cfg_dict.get("GRAPH_TRANSFORMER_OUT", 128))
        mol_dim = _infer_mol_graph_dim_from_checkpoint_cfg(cfg_dict)

        if mol_dim is not None:
            dist_adapter_only = abs(mol_dim - more_dim)
            dist_gt_concat = abs(mol_dim - (more_dim + gt_dim))
            if dist_adapter_only <= dist_gt_concat:
                cfg_dict["SOLUTE_ENCODER_TYPE"] = "graph_transformer_more_online_ln_adapter_no_gt"
            else:
                cfg_dict["SOLUTE_ENCODER_TYPE"] = "graph_transformer_more_online_ln_adapter_gt_concat"
        else:
            
            cfg_dict["SOLUTE_ENCODER_TYPE"] = "graph_transformer_more_online_ln_adapter_gt_concat"


def _resolve_existing_path(
    raw_path: Optional[str | Path],
    *,
    checkpoint_path: Optional[str | Path] = None,
    extra_roots: Sequence[Path] = (),
    required: bool = False,
    description: str = "",
) -> Optional[str]:
    if raw_path is None:
        if required:
            raise FileNotFoundError(f"{description}。")
        return None

    raw = str(raw_path).strip().strip('"').strip("'")
    if raw == "" or raw.lower() in {"none", "nan", "null"}:
        if required:
            raise FileNotFoundError(f"{description}。")
        return None

    raw = os.path.expandvars(os.path.expanduser(raw))
    p = Path(raw)

    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        for root in list(extra_roots) + _candidate_roots(str(checkpoint_path) if checkpoint_path else None):
            candidates.append(root / p)
        candidates.append(Path.cwd() / p)
        candidates.append(Path(__file__).resolve().parent / p)
        candidates.append(p)

    candidates = _unique_paths(candidates)
    for item in candidates:
        if item.exists():
            return str(item)

    legacy_name = p.name.lower()
    replacement_name: Optional[str] = None
    if legacy_name in {"development.csv", "flair_db.csv"} or re.fullmatch(
        r"(?:train|validation)_fold_\d+\.csv", p.name, flags=re.IGNORECASE
    ):
        replacement_name = "deployment.csv"
    elif legacy_name == "holdout_test.csv":
        replacement_name = "deployment_test.csv"

    if replacement_name is not None:
        fallback_candidates = [item.parent / replacement_name for item in candidates]
        fallback_candidates.extend(item.parent / "deployment" / replacement_name for item in candidates)
        for root in list(extra_roots) + _candidate_roots(str(checkpoint_path) if checkpoint_path else None):
            fallback_candidates.extend(
                [
                    root / replacement_name,
                    root / "data" / "splits" / replacement_name,
                    root / "data" / "splits" / "deployment" / replacement_name,
                ]
            )
        for item in _unique_paths(fallback_candidates):
            if item.exists():
                return str(item)

    if required:
        tried = "\n".join([f"  - {c}" for c in candidates])
        raise FileNotFoundError(
            f"{description}: {raw}\n：\n{tried}"
        )
    return str(candidates[0]) if candidates else raw


def _default_cfg_dict() -> dict:
    return {
        "TRAIN_CSV": "Train.csv",
        "VAL_CSV": "Val.csv",
        "TEST_CSV": "Test.csv",
        "CV_FOLD": None,
        "BATCH_SIZE": 1,
        "EPOCHS": 1,
        "MIN_EPOCHS": 1,
        "LR": 1e-3,
        "WEIGHT_DECAY": 1e-4,
        "L2SP_ENABLED": False,
        "L2SP_LAMBDA": 0.0,
        "L2SP_MODULES": ("mol_encoder.more_encoder",),
        "L2SP_EXCLUDE_BIAS_NORM": True,
        "DENOM_EPS": 1e-8,
        "SEED": 42,
        "NUM_WORKERS": 0,
        "PIN_MEMORY": False,
        "PERSISTENT_WORKERS": False,
        "PREFETCH_FACTOR": None,
        "CACHE_GRAPHS": False,
        "DROP_INVALID_SOLVENT_IN_GRAPH": True,
        "SOLUTE_ENCODER_TYPE": "graph_transformer",
        "MORE_WEIGHT_PATH": None,
        "MORE_NUM_LAYER": 5,
        "MORE_EMB_DIM": 300,
        "MORE_JK": "last",
        "MORE_DROPOUT_RATIO": 0.0,
        "MORE_GNN_TYPE": "gin",
        "MORE_POOLING": "mean",
        "MORE_FROZEN": True,
        "MORE_UNFROZEN_LAYERS": None,
        "SOLVENT_MODE": "morgan",
        "MORGAN_RADIUS": 4,
        "MORGAN_N_BITS": 256,
        "MORGAN_USE_CHIRALITY": False,
        "RDKIT_DESCRIPTOR_NAMES": None,
        "RDKIT_NAN_VALUE": 0.0,
        "RDKIT_INF_VALUE": 0.0,
        "TARGET_NAMES": ["abs", "emi", "plqy", "em"],
        "GRAPH_TRANSFORMER_HIDDEN": 128,
        "GRAPH_TRANSFORMER_LAYERS": 3,
        "GRAPH_TRANSFORMER_OUT": 128,
        "GRAPH_TRANSFORMER_HEADS": 4,
        "GRAPH_TRANSFORMER_EDGE_DIM": 6,
        "GRAPH_TRANSFORMER_DROPOUT": 0.1,
        "GRAPH_TRANSFORMER_USE_EDGE_ATTR": True,
        "GRAPH_TRANSFORMER_BETA": False,
        "GRAPH_TRANSFORMER_READOUT": "attention",
        "GRAPH_TRANSFORMER_READOUT_HEADS": None,
        "GRAPH_TRANSFORMER_READOUT_LAYERS": 1,
        "GRAPH_TRANSFORMER_READOUT_FFN_DIM": None,
        "GRAPH_TRANSFORMER_READOUT_DROPOUT": None,
        "SOLVENT_GRAPH_ENCODER_TYPE": "transformer",
        "SOLVENT_GAT_HIDDEN": 128,
        "SOLVENT_GAT_LAYERS": 3,
        "SOLVENT_GAT_OUT": 64,
        "SOLVENT_GAT_HEADS": 4,
        "SOLVENT_GAT_CONCAT": True,
        "SOLVENT_GAT_DROPOUT": 0.1,
        "SOLVENT_GRAPH_TRANSFORMER_HIDDEN": 128,
        "SOLVENT_GRAPH_TRANSFORMER_LAYERS": 3,
        "SOLVENT_GRAPH_TRANSFORMER_OUT": 128,
        "SOLVENT_GRAPH_TRANSFORMER_HEADS": 4,
        "SOLVENT_GRAPH_TRANSFORMER_EDGE_DIM": 6,
        "SOLVENT_GRAPH_TRANSFORMER_DROPOUT": 0.1,
        "SOLVENT_GRAPH_TRANSFORMER_USE_EDGE_ATTR": True,
        "SOLVENT_GRAPH_TRANSFORMER_BETA": False,
        "FUSION_TYPE": "concat",
        "FUSION_ATTENTION_DIM": 128,
        "FUSION_ATTENTION_HEADS": 4,
        "FUSION_ATTENTION_LAYERS": 1,
        "FUSION_ATTENTION_FFN_DIM": None,
        "FUSION_ATTENTION_DROPOUT": 0.1,
        "FUSION_OUTPUT_MODE": "cls",
        "MLP_HIDDEN": 256,
        "MLP_LAYERS": 3,
        "MLP_DROPOUT": 0.2,
        "EARLY_STOPPING_PATIENCE": 30,
        "EARLY_STOPPING_MIN_DELTA": 1e-4,
        "EARLY_STOPPING_MODE": "max",
        "EARLY_STOPPING_VERBOSE": False,
        "LR_SCHEDULER_ENABLED": False,
        "LR_SCHEDULER_NAME": "ReduceLROnPlateau",
        "LR_SCHEDULER_MONITOR": "val_loss",
        "LR_SCHEDULER_MODE": "min",
        "LR_SCHEDULER_FACTOR": 0.5,
        "LR_SCHEDULER_PATIENCE": 10,
        "LR_SCHEDULER_THRESHOLD": 1e-4,
        "LR_SCHEDULER_THRESHOLD_MODE": "rel",
        "LR_SCHEDULER_COOLDOWN": 0,
        "LR_SCHEDULER_MIN_LR": 1e-6,
        "LR_SCHEDULER_EPS": 1e-8,
        "FULL_TRAIN_EVAL_EACH_EPOCH": False,
        "RUN_IDX": 1,
    }


def _cfg_from_checkpoint(ckpt: dict, checkpoint_path: str, device: torch.device) -> SimpleNamespace:
    cfg_dict = _default_cfg_dict()
    raw_cfg = ckpt.get("cfg", {})
    if isinstance(raw_cfg, SimpleNamespace):
        raw_cfg = vars(raw_cfg)
    if isinstance(raw_cfg, dict):
        cfg_dict.update(raw_cfg)
    else:
        raise ValueError("checkpoint  cfg  dict / SimpleNamespace，。")

    _infer_architecture_from_checkpoint_state(ckpt, cfg_dict, raw_cfg)

    if _cfg_value_missing(cfg_dict, "CV_FOLD"):
        for value in (raw_cfg.get("TRAIN_CSV"), raw_cfg.get("VAL_CSV"), checkpoint_path):
            match = re.search(r"fold[_ -]*0*(\d+)", str(value or ""), flags=re.IGNORECASE)
            if match:
                cfg_dict["CV_FOLD"] = int(match.group(1))
                break

    roots = _candidate_roots(checkpoint_path)

    
    cfg_dict["DEVICE"] = device
    cfg_dict["BATCH_SIZE"] = 1
    cfg_dict["NUM_WORKERS"] = 0
    cfg_dict["PIN_MEMORY"] = False
    cfg_dict["PERSISTENT_WORKERS"] = False
    cfg_dict["PREFETCH_FACTOR"] = None
    cfg_dict["CACHE_GRAPHS"] = False
    cfg_dict["ALLOW_MISSING_MORE_WEIGHT_FOR_CHECKPOINT_LOAD"] = bool(
        cfg_dict.get("ALLOW_MISSING_MORE_WEIGHT_FOR_CHECKPOINT_LOAD", False)
    )

    
    cfg_dict["TRAIN_CSV"] = _resolve_existing_path(
        cfg_dict.get("TRAIN_CSV"),
        checkpoint_path=checkpoint_path,
        extra_roots=roots,
        required=False,
        description=" TRAIN_CSV",
    )
    cfg_dict["VAL_CSV"] = _resolve_existing_path(
        cfg_dict.get("VAL_CSV"),
        checkpoint_path=checkpoint_path,
        extra_roots=roots,
        required=False,
        description=" VAL_CSV",
    )
    cfg_dict["TEST_CSV"] = _resolve_existing_path(
        cfg_dict.get("TEST_CSV"),
        checkpoint_path=checkpoint_path,
        extra_roots=roots,
        required=False,
        description=" TEST_CSV",
    )

    if cfg_dict.get("MORE_WEIGHT_PATH") not in {None, "", "None"}:
        resolved_more_path = _resolve_existing_path(
            cfg_dict.get("MORE_WEIGHT_PATH"),
            checkpoint_path=checkpoint_path,
            extra_roots=roots,
            required=not _checkpoint_has_more_state(ckpt),
            description="MORE  MORE_WEIGHT_PATH",
        )
        if resolved_more_path is not None and Path(str(resolved_more_path)).exists():
            cfg_dict["MORE_WEIGHT_PATH"] = resolved_more_path
        elif _checkpoint_has_more_state(ckpt):
            
            cfg_dict["MORE_WEIGHT_PATH"] = None

    if cfg_dict.get("GRAPH_TRANSFORMER_READOUT_HEADS") is None:
        cfg_dict["GRAPH_TRANSFORMER_READOUT_HEADS"] = cfg_dict.get("GRAPH_TRANSFORMER_HEADS", 4)
    if cfg_dict.get("GRAPH_TRANSFORMER_READOUT_DROPOUT") is None:
        cfg_dict["GRAPH_TRANSFORMER_READOUT_DROPOUT"] = cfg_dict.get("GRAPH_TRANSFORMER_DROPOUT", 0.1)

    return SimpleNamespace(**cfg_dict)







def _target_stats_from_checkpoint(ckpt: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    mean = None
    std = None

    for key in ("target_mean", "TARGET_MEAN", "y_mean", "Y_MEAN"):
        if key in ckpt:
            mean = _to_numpy_1d(ckpt[key])
            break
    for key in ("target_std", "TARGET_STD", "y_std", "Y_STD"):
        if key in ckpt:
            std = _to_numpy_1d(ckpt[key])
            break

    if mean is None or std is None:
        return None, None
    if mean.shape != std.shape:
        raise ValueError(f"checkpoint  target_mean/target_std : {mean.shape} vs {std.shape}")
    return mean.astype(np.float32), std.astype(np.float32)


def _load_target_stats(dl, cfg: SimpleNamespace, ckpt: dict, checkpoint_path: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    mean, std = _target_stats_from_checkpoint(ckpt)

    train_csv = getattr(cfg, "TRAIN_CSV", None)
    if train_csv is None or not str(train_csv).strip() or not Path(str(train_csv)).exists():
        if mean is not None and std is not None:
            
            dummy = pd.DataFrame()
            return mean, std, dummy
        raise FileNotFoundError(
            " checkpoint  target_mean / target_std， cfg.TRAIN_CSV。\n"
            " data/splits/deployment/deployment.csv ， MAIN_MODEL_ROOT。\n"
            f"checkpoint: {checkpoint_path}\n"
            f"cfg.TRAIN_CSV: {train_csv}"
        )

    train_df = dl.build_manifest_df(
        csv_path=str(train_csv),
        solvent_mode=getattr(cfg, "SOLVENT_MODE", "morgan"),
        drop_invalid_solvent_in_graph=bool(getattr(cfg, "DROP_INVALID_SOLVENT_IN_GRAPH", True)),
        cv_fold=getattr(cfg, "CV_FOLD", None),
        fold_role="train" if getattr(cfg, "CV_FOLD", None) is not None else None,
    )

    if mean is None or std is None:
        mean, std = dl.compute_target_stats(train_df)

    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean.shape != std.shape:
        raise ValueError(f"target_mean/target_std : {mean.shape} vs {std.shape}")
    if mean.size != len(getattr(dl, "TARGET_COLS", ["abs", "emi", "plqy", "em"])):
        raise ValueError(
            f"target_mean/target_std  {mean.size}， TARGET_COLS  "
            f"{getattr(dl, 'TARGET_COLS', None)}。"
        )
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    return mean, std, train_df


def _make_inference_df(dl, smiles: str, solvent: Optional[str]) -> pd.DataFrame:
    solvent_text = "" if solvent is None else str(solvent).strip()
    target_cols = list(getattr(dl, "TARGET_COLS", ["abs", "emi", "plqy", "em"]))

    row = {
        getattr(dl, "SAMPLE_ID_COL", "sample_id"): 0,
        getattr(dl, "RAW_ROW_ID_COL", "raw_row_id"): 0,
        getattr(dl, "SMILES_COL", "smiles"): str(smiles).strip(),
        getattr(dl, "SOLVENT_COL", "solvent"): solvent_text,
        getattr(dl, "SOLVENT_RAW_COL", "solvent_raw"): solvent_text,
    }
    for col in target_cols:
        row[col] = np.nan
    return pd.DataFrame([row])


def _dataset_kwargs(cfg: SimpleNamespace) -> dict:
    return {
        "solvent_mode": getattr(cfg, "SOLVENT_MODE", "morgan"),
        "drop_invalid_solvent_in_graph": bool(getattr(cfg, "DROP_INVALID_SOLVENT_IN_GRAPH", True)),
        "cache_graphs": False,
        "cgsd_csv_path": getattr(cfg, "CGSD_CSV_PATH", None),
        "morgan_kwargs": {
            "radius": int(getattr(cfg, "MORGAN_RADIUS", 4)),
            "n_bits": int(getattr(cfg, "MORGAN_N_BITS", 256)),
            "use_chirality": bool(getattr(cfg, "MORGAN_USE_CHIRALITY", False)),
        },
        "rdkit_kwargs": {
            "descriptor_names": getattr(cfg, "RDKIT_DESCRIPTOR_NAMES", None),
            "nan_value": float(getattr(cfg, "RDKIT_NAN_VALUE", 0.0)),
            "inf_value": float(getattr(cfg, "RDKIT_INF_VALUE", 0.0)),
        },
    }


def _make_single_batch(dl, PyGDataLoader, *, smiles: str, solvent: Optional[str], cfg: SimpleNamespace, target_mean, target_std):
    df = _make_inference_df(dl, smiles=smiles, solvent=solvent)
    ds = dl.FluorSolventDataset(
        df=df,
        target_mean=target_mean,
        target_std=target_std,
        **_dataset_kwargs(cfg),
    )
    loader = PyGDataLoader(ds, batch_size=1, shuffle=False, follow_batch=["solvent_x"])
    return next(iter(loader))


def _make_sample_batch_for_build_system(dl, PyGDataLoader, *, cfg: SimpleNamespace, train_df: pd.DataFrame, target_mean, target_std):
    if train_df is not None and len(train_df) > 0:
        smi_col = getattr(dl, "SMILES_COL", "smiles")
        sol_col = getattr(dl, "SOLVENT_COL", "solvent")
        row = train_df.iloc[0]
        smiles = str(row[smi_col])
        solvent = str(row[sol_col]) if sol_col in train_df.columns and not pd.isna(row[sol_col]) else ""
    else:
        
        smiles = "CCO"
        solvent = "O"
    return _make_single_batch(
        dl,
        PyGDataLoader,
        smiles=smiles,
        solvent=solvent,
        cfg=cfg,
        target_mean=target_mean,
        target_std=target_std,
    )







def _strip_module_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if not keys or not all(str(k).startswith("module.") for k in keys):
        return state_dict
    return {str(k)[7:]: v for k, v in state_dict.items()}


def _load_state(module, state_dict, name: str, strict: bool = True) -> None:
    if state_dict is None:
        raise KeyError(f"checkpoint  {name} 。")
    try:
        module.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        module.load_state_dict(_strip_module_prefix(state_dict), strict=strict)


def _set_eval(system: SimpleNamespace) -> None:
    for name in ("mol_encoder", "sol_encoder", "fusion_encoder", "mlp"):
        module = getattr(system, name, None)
        if hasattr(module, "eval"):
            module.eval()







def _cfg_get(name: str, default=None):
    """ reliability config.py ； default。"""
    try:
        import config as reliability_config
        return getattr(reliability_config, name, default)
    except Exception:
        return default


def _set_if_not_none(cfg_dict: dict, key: str, value) -> None:
    if value is not None:
        cfg_dict[key] = value


def _make_training_cfg(train_csv: str, val_csv: str, test_csv: str, fold_idx: int, model_dir: str) -> SimpleNamespace:
    """ fold  cfg。

     config.CV_BASE_CHECKPOINT  cfg ；，
    model_interface.py ， config.py  CV_* 。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ckpt_path = _cfg_get("CV_BASE_CHECKPOINT", None)

    if base_ckpt_path is not None and str(base_ckpt_path).strip() not in {"", "None", "none", "null"}:
        resolved_base = _resolve_existing_path(
            base_ckpt_path,
            checkpoint_path=None,
            extra_roots=_candidate_roots(None),
            required=True,
            description="CV_BASE_CHECKPOINT",
        )
        ckpt = _torch_load(resolved_base, map_location=device)
        if not isinstance(ckpt, dict) or "cfg" not in ckpt:
            raise ValueError(f"CV_BASE_CHECKPOINT  cfg  checkpoint: {resolved_base}")
        
        cfg_dict = vars(_cfg_from_checkpoint(ckpt, checkpoint_path=resolved_base, device=device)).copy()
    else:
        cfg_dict = _default_cfg_dict()
        cfg_dict["DEVICE"] = device

    
    split_type = str(_cfg_get("CV_SPLIT_TYPE", "random")).strip().lower()
    fold_dir = Path(model_dir) / f"{split_type}_5fold" / f"fold_{int(fold_idx)}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict["TRAIN_CSV"] = str(Path(train_csv).resolve())
    cfg_dict["VAL_CSV"] = str(Path(val_csv).resolve())
    cfg_dict["TEST_CSV"] = str(Path(test_csv).resolve())
    cfg_dict["BEST_PATH"] = str(fold_dir / "best_model.pt")
    cfg_dict["LOG_PATH"] = str(fold_dir / "train_log.txt")
    cfg_dict["LOG_CSV_PATH"] = str(fold_dir / "train_log.csv")
    cfg_dict["RUN_IDX"] = int(fold_idx) + 1
    cfg_dict["CV_FOLD"] = int(fold_idx) + 1
    cfg_dict["TARGET_NAMES"] = ["abs", "emi", "plqy", "em"]

    
    override_map = {
        "EPOCHS": ("CV_EPOCHS", cfg_dict.get("EPOCHS", 2000)),
        "MIN_EPOCHS": ("CV_MIN_EPOCHS", cfg_dict.get("MIN_EPOCHS", 70)),
        "BATCH_SIZE": ("CV_BATCH_SIZE", cfg_dict.get("BATCH_SIZE", 256)),
        "LR": ("CV_LR", cfg_dict.get("LR", 1e-3)),
        "WEIGHT_DECAY": ("CV_WEIGHT_DECAY", cfg_dict.get("WEIGHT_DECAY", 1e-4)),
        "NUM_WORKERS": ("CV_NUM_WORKERS", cfg_dict.get("NUM_WORKERS", 0)),
        "PIN_MEMORY": ("CV_PIN_MEMORY", cfg_dict.get("PIN_MEMORY", None)),
        "PERSISTENT_WORKERS": ("CV_PERSISTENT_WORKERS", cfg_dict.get("PERSISTENT_WORKERS", None)),
        "PREFETCH_FACTOR": ("CV_PREFETCH_FACTOR", cfg_dict.get("PREFETCH_FACTOR", 2)),
        "CACHE_GRAPHS": ("CV_CACHE_GRAPHS", cfg_dict.get("CACHE_GRAPHS", True)),
        "DROP_INVALID_SOLVENT_IN_GRAPH": (
            "CV_DROP_INVALID_SOLVENT_IN_GRAPH",
            cfg_dict.get("DROP_INVALID_SOLVENT_IN_GRAPH", True),
        ),
        "EARLY_STOPPING_PATIENCE": (
            "CV_EARLY_STOPPING_PATIENCE",
            cfg_dict.get("EARLY_STOPPING_PATIENCE", 30),
        ),
        "L2SP_ENABLED": ("CV_L2SP_ENABLED", cfg_dict.get("L2SP_ENABLED", False)),
        "L2SP_LAMBDA": ("CV_L2SP_LAMBDA", cfg_dict.get("L2SP_LAMBDA", 0.0)),
    }
    for key, (cv_name, default_value) in override_map.items():
        _set_if_not_none(cfg_dict, key, _cfg_get(cv_name, default_value))

    
    arch_override_map = {
        "SOLUTE_ENCODER_TYPE": "CV_SOLUTE_ENCODER_TYPE",
        "MORE_WEIGHT_PATH": "CV_MORE_WEIGHT_PATH",
        "MORE_NUM_LAYER": "CV_MORE_NUM_LAYER",
        "MORE_EMB_DIM": "CV_MORE_EMB_DIM",
        "MORE_JK": "CV_MORE_JK",
        "MORE_DROPOUT_RATIO": "CV_MORE_DROPOUT_RATIO",
        "MORE_GNN_TYPE": "CV_MORE_GNN_TYPE",
        "MORE_POOLING": "CV_MORE_POOLING",
        "MORE_FROZEN": "CV_MORE_FROZEN",
        "MORE_UNFROZEN_LAYERS": "CV_MORE_UNFROZEN_LAYERS",
        "SOLVENT_MODE": "CV_SOLVENT_MODE",
        "MORGAN_RADIUS": "CV_MORGAN_RADIUS",
        "MORGAN_N_BITS": "CV_MORGAN_N_BITS",
        "MORGAN_USE_CHIRALITY": "CV_MORGAN_USE_CHIRALITY",
        "SOLVENT_GRAPH_ENCODER_TYPE": "CV_SOLVENT_GRAPH_ENCODER_TYPE",
        "FUSION_TYPE": "CV_FUSION_TYPE",
        "MLP_HIDDEN": "CV_MLP_HIDDEN",
        "MLP_LAYERS": "CV_MLP_LAYERS",
        "MLP_DROPOUT": "CV_MLP_DROPOUT",
        "GRAPH_TRANSFORMER_HIDDEN": "CV_GRAPH_TRANSFORMER_HIDDEN",
        "GRAPH_TRANSFORMER_LAYERS": "CV_GRAPH_TRANSFORMER_LAYERS",
        "GRAPH_TRANSFORMER_OUT": "CV_GRAPH_TRANSFORMER_OUT",
        "GRAPH_TRANSFORMER_HEADS": "CV_GRAPH_TRANSFORMER_HEADS",
        "GRAPH_TRANSFORMER_DROPOUT": "CV_GRAPH_TRANSFORMER_DROPOUT",
        "GRAPH_TRANSFORMER_READOUT": "CV_GRAPH_TRANSFORMER_READOUT",
        "GRAPH_TRANSFORMER_READOUT_HEADS": "CV_GRAPH_TRANSFORMER_READOUT_HEADS",
        "GRAPH_TRANSFORMER_READOUT_LAYERS": "CV_GRAPH_TRANSFORMER_READOUT_LAYERS",
        "GRAPH_TRANSFORMER_READOUT_FFN_DIM": "CV_GRAPH_TRANSFORMER_READOUT_FFN_DIM",
        "GRAPH_TRANSFORMER_READOUT_DROPOUT": "CV_GRAPH_TRANSFORMER_READOUT_DROPOUT",
        "SOLVENT_GRAPH_TRANSFORMER_HIDDEN": "CV_SOLVENT_GRAPH_TRANSFORMER_HIDDEN",
        "SOLVENT_GRAPH_TRANSFORMER_LAYERS": "CV_SOLVENT_GRAPH_TRANSFORMER_LAYERS",
        "SOLVENT_GRAPH_TRANSFORMER_OUT": "CV_SOLVENT_GRAPH_TRANSFORMER_OUT",
        "SOLVENT_GRAPH_TRANSFORMER_HEADS": "CV_SOLVENT_GRAPH_TRANSFORMER_HEADS",
        "SOLVENT_GRAPH_TRANSFORMER_DROPOUT": "CV_SOLVENT_GRAPH_TRANSFORMER_DROPOUT",
        "FUSION_ATTENTION_DIM": "CV_FUSION_ATTENTION_DIM",
        "FUSION_ATTENTION_HEADS": "CV_FUSION_ATTENTION_HEADS",
        "FUSION_ATTENTION_LAYERS": "CV_FUSION_ATTENTION_LAYERS",
        "FUSION_ATTENTION_FFN_DIM": "CV_FUSION_ATTENTION_FFN_DIM",
        "FUSION_ATTENTION_DROPOUT": "CV_FUSION_ATTENTION_DROPOUT",
        "FUSION_OUTPUT_MODE": "CV_FUSION_OUTPUT_MODE",
    }
    for key, cv_name in arch_override_map.items():
        _set_if_not_none(cfg_dict, key, _cfg_get(cv_name, None))

    
    cfg_dict["DEVICE"] = device
    cfg_dict["DENOM_EPS"] = float(cfg_dict.get("DENOM_EPS", 1e-8))
    cfg_dict["SEED"] = int(_cfg_get("RANDOM_SEED", cfg_dict.get("SEED", 42))) + int(fold_idx)
    cfg_dict["EARLY_STOPPING_PATIENCE"] = int(cfg_dict.get("EARLY_STOPPING_PATIENCE", 30))
    cfg_dict["EARLY_STOPPING_MIN_DELTA"] = float(cfg_dict.get("EARLY_STOPPING_MIN_DELTA", 1e-4))
    cfg_dict["EARLY_STOPPING_MODE"] = str(cfg_dict.get("EARLY_STOPPING_MODE", "max"))
    cfg_dict["EARLY_STOPPING_VERBOSE"] = bool(cfg_dict.get("EARLY_STOPPING_VERBOSE", False))
    if int(cfg_dict["EPOCHS"]) < int(cfg_dict["MIN_EPOCHS"]):
        cfg_dict["MIN_EPOCHS"] = int(cfg_dict["EPOCHS"])

    return SimpleNamespace(**cfg_dict)


def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))






class MyPredictionModel:
    """
    。

    ：
    - MyPredictionModel.load(best_path)
    - model.predict(solute_mol, solvent_mol, smiles, solvent)

    fit / save ；scaffold  cv ， scaffold training_info.pkl  train_reliability / evaluate。
    """

    def __init__(
        self,
        *,
        system: Optional[SimpleNamespace] = None,
        cfg: Optional[SimpleNamespace] = None,
        target_mean: Optional[np.ndarray] = None,
        target_std: Optional[np.ndarray] = None,
        target_names: Optional[Sequence[str]] = None,
        dl=None,
        tr=None,
        PyGDataLoader=None,
        checkpoint_path: Optional[str] = None,
    ):
        self.system = system
        self.cfg = cfg
        self.target_mean = _to_numpy_1d(target_mean) if target_mean is not None else None
        self.target_std = _to_numpy_1d(target_std) if target_std is not None else None
        self.target_names = list(target_names) if target_names is not None else ["abs", "emi", "plqy", "em"]
        self.dl = dl
        self.tr = tr
        self.PyGDataLoader = PyGDataLoader
        self.checkpoint_path = checkpoint_path

    def fit(self, train_records, y_train: Dict[str, object], fold_idx: Optional[int] = None):
        raise NotImplementedError(
            " fit(train_records, y_train)。 "
            "MyPredictionModel.train_from_csv(train_csv, val_csv, fold_idx, model_dir)，"
            " data_loading/model_factory/trainer  DataLoader 。"
        )

    @classmethod
    def train_from_csv(
        cls,
        train_csv: str,
        val_csv: str,
        fold_idx: int,
        model_dir: str = "models",
        test_csv: Optional[str] = None,
    ):
        """ fold， checkpoint 。

        ；/ config.CV_BASE_CHECKPOINT  config.py  CV_* 。
        """
        dl, mf, tr, _ = _import_main_modules(None)
        cfg = _make_training_cfg(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv or val_csv,
            fold_idx=fold_idx,
            model_dir=model_dir,
        )
        _set_global_seed(int(getattr(cfg, "SEED", 42)))
        _patch_more_only_encoder_if_needed(mf, getattr(cfg, "SOLUTE_ENCODER_TYPE", "graph_transformer"))

        train_loader, val_loader, test_loader, _ = dl.make_train_val_test_loaders(
            train_csv_path=cfg.TRAIN_CSV,
            val_csv_path=cfg.VAL_CSV,
            test_csv_path=cfg.TEST_CSV,
            solvent_mode=getattr(cfg, "SOLVENT_MODE", "morgan"),
            batch_size=int(getattr(cfg, "BATCH_SIZE", 256)),
            seed=int(getattr(cfg, "SEED", 42)),
            num_workers=int(getattr(cfg, "NUM_WORKERS", 0)),
            drop_invalid_solvent_in_graph=bool(getattr(cfg, "DROP_INVALID_SOLVENT_IN_GRAPH", True)),
            pin_memory=getattr(cfg, "PIN_MEMORY", None),
            persistent_workers=getattr(cfg, "PERSISTENT_WORKERS", None),
            prefetch_factor=getattr(cfg, "PREFETCH_FACTOR", 2),
            cache_graphs=bool(getattr(cfg, "CACHE_GRAPHS", True)),
            cgsd_csv_path=getattr(cfg, "CGSD_CSV_PATH", None),
            morgan_kwargs={
                "radius": int(getattr(cfg, "MORGAN_RADIUS", 4)),
                "n_bits": int(getattr(cfg, "MORGAN_N_BITS", 256)),
                "use_chirality": bool(getattr(cfg, "MORGAN_USE_CHIRALITY", False)),
            },
            rdkit_kwargs={
                "descriptor_names": getattr(cfg, "RDKIT_DESCRIPTOR_NAMES", None),
                "nan_value": float(getattr(cfg, "RDKIT_NAN_VALUE", 0.0)),
                "inf_value": float(getattr(cfg, "RDKIT_INF_VALUE", 0.0)),
            },
            cv_fold=getattr(cfg, "CV_FOLD", None),
        )

        sample_batch = next(iter(train_loader)).to(cfg.DEVICE)
        system = mf.build_system(sample_batch=sample_batch, cfg=cfg)
        metrics = tr.train_experiment(system, train_loader, val_loader, test_loader, cfg)
        model = cls.load(
            cfg.BEST_PATH,
            train_csv_override=cfg.TRAIN_CSV,
            cv_fold_override=getattr(cfg, "CV_FOLD", None),
        )
        model.training_metrics = metrics
        return model, str(cfg.BEST_PATH), metrics, cfg

    @classmethod
    def load(
        cls,
        path: str,
        train_csv_override: Optional[str] = None,
        cv_fold_override: Optional[int] = None,
    ):
        checkpoint_path = str(Path(path).expanduser())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _log(f"[MainModel] loading checkpoint: {checkpoint_path}")
        _log(f"[MainModel] device: {device}")

        dl, mf, tr, PyGDataLoader = _import_main_modules(checkpoint_path)
        _log("[MainModel] main model modules imported")
        ckpt = _torch_load(checkpoint_path, map_location=device)
        if not isinstance(ckpt, dict):
            raise ValueError(f"{checkpoint_path}  dict checkpoint，。")
        if "cfg" not in ckpt:
            raise ValueError(f"{checkpoint_path}  cfg，。")

        cfg = _cfg_from_checkpoint(ckpt, checkpoint_path=checkpoint_path, device=device)
        _log("[MainModel] checkpoint cfg resolved")
        if train_csv_override is not None:
            cfg.TRAIN_CSV = _resolve_existing_path(
                train_csv_override,
                checkpoint_path=checkpoint_path,
                extra_roots=_candidate_roots(checkpoint_path),
                required=True,
                description=" train_csv_override",
            )
        if cv_fold_override is not None:
            cfg.CV_FOLD = int(cv_fold_override)
        more_path_missing = getattr(cfg, "MORE_WEIGHT_PATH", None) in {None, "", "None", "none", "null"}
        fallback_more_path = _cfg_get("CV_MORE_WEIGHT_PATH", None)
        if more_path_missing and fallback_more_path not in {None, "", "None", "none", "null"}:
            cfg.MORE_WEIGHT_PATH = _resolve_existing_path(
                fallback_more_path,
                checkpoint_path=checkpoint_path,
                extra_roots=_candidate_roots(checkpoint_path),
                required=True,
                description=" CV_MORE_WEIGHT_PATH",
            )
        _patch_more_only_encoder_if_needed(mf, getattr(cfg, "SOLUTE_ENCODER_TYPE", "graph_transformer"))

        _log("[MainModel] loading target normalization stats")
        target_mean, target_std, train_df = _load_target_stats(dl, cfg, ckpt, checkpoint_path)
        _log("[MainModel] building sample batch")
        sample_batch = _make_sample_batch_for_build_system(
            dl,
            PyGDataLoader,
            cfg=cfg,
            train_df=train_df,
            target_mean=target_mean,
            target_std=target_std,
        )
        sample_batch = sample_batch.to(device)

        _log("[MainModel] building model system")
        system = mf.build_system(sample_batch=sample_batch, cfg=cfg)
        _log("[MainModel] loading checkpoint state_dict")

        _load_state(system.mol_encoder, ckpt.get("mol_encoder"), "mol_encoder", strict=True)
        _load_state(system.mlp, ckpt.get("mlp"), "mlp", strict=True)

        if "sol_encoder" in ckpt and ckpt.get("sol_encoder") is not None:
            
            _load_state(system.sol_encoder, ckpt.get("sol_encoder"), "sol_encoder", strict=False)

        if getattr(system, "use_fusion_self_attention", False):
            if ckpt.get("fusion_encoder") is None:
                raise KeyError("checkpoint  fusion_encoder， cfg  fusion self-attention。")
            _load_state(system.fusion_encoder, ckpt.get("fusion_encoder"), "fusion_encoder", strict=False)

        _set_eval(system)
        _log("[MainModel] checkpoint ready")

        target_names = list(getattr(cfg, "TARGET_NAMES", getattr(dl, "TARGET_COLS", ["abs", "emi", "plqy", "em"])))
        return cls(
            system=system,
            cfg=cfg,
            target_mean=target_mean,
            target_std=target_std,
            target_names=target_names,
            dl=dl,
            tr=tr,
            PyGDataLoader=PyGDataLoader,
            checkpoint_path=checkpoint_path,
        )

    def predict(
        self,
        solute_mol: Chem.Mol,
        solvent_mol: Optional[Chem.Mol] = None,
        smiles: Optional[str] = None,
        solvent: Optional[str] = None,
    ) -> Dict[str, float]:
        if self.system is None or self.cfg is None:
            raise RuntimeError("。 MyPredictionModel.load(path)。")
        if self.dl is None or self.tr is None or self.PyGDataLoader is None:
            raise RuntimeError("。")

        if smiles is None and solute_mol is not None:
            smiles = Chem.MolToSmiles(solute_mol)
        if solvent is None and solvent_mol is not None:
            solvent = Chem.MolToSmiles(solvent_mol)

        smiles = "" if smiles is None else str(smiles).strip()
        solvent = None if solvent is None else str(solvent).strip()
        if smiles == "":
            return {p: np.nan for p in PROPERTIES}

        batch = _make_single_batch(
            self.dl,
            self.PyGDataLoader,
            smiles=smiles,
            solvent=solvent,
            cfg=self.cfg,
            target_mean=self.target_mean,
            target_std=self.target_std,
        )
        batch = batch.to(self.system.device)

        _set_eval(self.system)
        with torch.no_grad():
            x_vec = self.tr.build_x_vec(batch, self.system)
            y_scaled = self.system.mlp(x_vec)

        pred_scaled = y_scaled.detach().cpu().numpy().reshape(-1).astype(np.float32)
        target_mean = np.asarray(self.target_mean, dtype=np.float32).reshape(-1)
        target_std = np.asarray(self.target_std, dtype=np.float32).reshape(-1)

        if pred_scaled.size != target_mean.size:
            raise ValueError(
                f"={pred_scaled.size}，target_mean/std ={target_mean.size}，。"
            )

        pred_raw = pred_scaled * target_std + target_mean

        pred_by_name: Dict[str, float] = {}
        for i, name in enumerate(self.target_names):
            if i >= pred_raw.size:
                break
            pred_by_name[str(name).lower()] = float(pred_raw[i])

        
        return {
            "plqy": float(pred_by_name.get("plqy", np.nan)),
            "emi": float(pred_by_name.get("emi", np.nan)),
            "em": float(pred_by_name.get("em", np.nan)),
            "abs": float(pred_by_name.get("abs", np.nan)),
        }

    def save(self, path: str):
        raise NotImplementedError(
            "； trainer.py  best_model.pt。"
        )



DummyMultiTaskModel = MyPredictionModel
MyGNNModel = MyPredictionModel


def predict_with_model(model, record: Dict) -> Dict[str, float]:
    """。"""
    solute_mol = record.get("solute_mol")
    solvent_mol = record.get("solvent_mol")
    smiles = record.get("smiles")
    solvent = record.get("solvent")

    return model.predict(
        solute_mol=solute_mol,
        solvent_mol=solvent_mol,
        smiles=smiles,
        solvent=solvent,
    )

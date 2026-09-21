from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

CODE_ROOT = Path(__file__).parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(CODE_ROOT)

import model_factory as _model_factory
from experiment_runner import RunConfig, RunIOConfig, run_from_io_config


class GraphTransformerOnlineMOREAdapterOnlyEncoder(nn.Module):
    """Use frozen online MORE as mol_vec; keep solute GT atom tokens for fusion."""

    def __init__(self, graph_encoder, more_encoder):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.more_encoder = more_encoder
        self.more_adapter = _model_factory.MORELayerNormAdapter(in_dim=int(more_encoder.graph_dim))
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


_ORIGINAL_BUILD_MOL_ENCODER = _model_factory._build_mol_encoder
_MORE_ONLY_PATCHED = False


def _build_mol_encoder_more_only(sample_batch, cfg):
    raw_encoder_type = getattr(cfg, "SOLUTE_ENCODER_TYPE", "graph_transformer")
    encoder_type = str(raw_encoder_type).strip().lower()
    alias = {
        "graph_transformer_more_online_ln_adapter_only": "graph_transformer_more_online_ln_adapter_only",
        "graph_transformer_online_more_ln_adapter_only": "graph_transformer_more_online_ln_adapter_only",
        "online_more_ln_adapter_only": "graph_transformer_more_online_ln_adapter_only",
        "more_online_ln_adapter_only": "graph_transformer_more_online_ln_adapter_only",
        "graph_transformer_more_online_ln_adapter_no_gt": "graph_transformer_more_online_ln_adapter_only",
        "graph_transformer_online_more_ln_adapter_no_gt": "graph_transformer_more_online_ln_adapter_only",
        "online_more_ln_adapter_no_gt": "graph_transformer_more_online_ln_adapter_only",
        "more_online_ln_adapter_no_gt": "graph_transformer_more_online_ln_adapter_only",
    }
    encoder_type = alias.get(encoder_type, encoder_type)

    if encoder_type != "graph_transformer_more_online_ln_adapter_only":
        return _ORIGINAL_BUILD_MOL_ENCODER(sample_batch, cfg)

    graph_encoder, _ = _model_factory._build_graph_transformer_encoder(sample_batch, cfg)
    more_weight_path = getattr(cfg, "MORE_WEIGHT_PATH", None)
    if more_weight_path is None or str(more_weight_path).strip() == "":
        raise ValueError("no-GT MORE adapter  MORE_WEIGHT_PATH， 'MORE.pth'。")

    more_encoder = _model_factory.OnlineMOREGraphEncoder(
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

    mol_encoder = GraphTransformerOnlineMOREAdapterOnlyEncoder(
        graph_encoder=graph_encoder,
        more_encoder=more_encoder,
    ).to(cfg.DEVICE)
    return mol_encoder, int(mol_encoder.graph_dim)


def _patch_more_only_encoder_if_needed(solute_encoder_type: str) -> None:
    global _MORE_ONLY_PATCHED
    key = str(solute_encoder_type).strip().lower()
    if "adapter_only" not in key and "no_gt" not in key:
        return
    if not _MORE_ONLY_PATCHED:
        _model_factory._build_mol_encoder = _build_mol_encoder_more_only
        _MORE_ONLY_PATCHED = True


BASE_CONFIG = {
    "output_root": "HPO_frozenMORE_base",

    
    "more_weight_path": "../../models/pretrained/MORE.pth",
    "more_num_layer": 5,
    "more_emb_dim": 300,
    "more_jk": "last",
    "more_dropout_ratio": 0.0,
    "more_gnn_type": "gin",
    "more_pooling": "mean",
    "more_frozen": True,
    "more_unfrozen_layers": 0,

    "solute_encoder_type": "graph_transformer_more_online_ln_adapter_gt_concat",

    "seed": 42,
    "batch_size": 256,
    "epochs": 2000,
    "min_epochs": 70,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 30,

    
    "l2sp_enabled": False,
    "l2sp_lambda": 0.0,
    "l2sp_modules": ("mol_encoder.more_encoder",),
    "l2sp_exclude_bias_norm": True,

    "num_workers": 4,
    "cache_graphs": True,
    "drop_invalid_solvent_in_graph": True,

    
    "graph_transformer_hidden": 128,
    "graph_transformer_layers": 3,
    "graph_transformer_out": 128,
    "graph_transformer_heads": 4,
    "graph_transformer_edge_dim": 6,
    "graph_transformer_dropout": 0.1,
    "graph_transformer_use_edge_attr": True,
    "graph_transformer_beta": False,
    "graph_transformer_readout": "attention",
    "graph_transformer_readout_heads": None,
    "graph_transformer_readout_layers": 1,
    "graph_transformer_readout_ffn_dim": None,
    "graph_transformer_readout_dropout": None,

    
    "solvent_mode": "morgan",
    "morgan_radius": 4,
    "morgan_n_bits": 256,
    "morgan_use_chirality": False,

    
    "solvent_graph_encoder_type": "transformer",
    "solvent_graph_transformer_hidden": 128,
    "solvent_graph_transformer_layers": 3,
    "solvent_graph_transformer_out": 128,
    "solvent_graph_transformer_heads": 4,
    "solvent_graph_transformer_edge_dim": 6,
    "solvent_graph_transformer_dropout": 0.1,
    "solvent_graph_transformer_use_edge_attr": True,
    "solvent_graph_transformer_beta": False,

    
    "fusion_type": "self_attention",
    "fusion_attention_dim": 128,
    "fusion_attention_heads": 4,
    "fusion_attention_layers": 1,
    "fusion_attention_ffn_dim": None,
    "fusion_attention_dropout": 0.1,
    "fusion_output_mode": "cls",

    
    "mlp_hidden": 256,
    "mlp_layers": 3,
    "mlp_dropout": 0.2,
}


RUNS_BY_SPLIT = {}


def _merged_config(variant_config: dict) -> dict:
    cfg = dict(BASE_CONFIG)
    cfg.update(variant_config)
    cfg["more_frozen"] = True
    cfg["more_unfrozen_layers"] = 0
    cfg["l2sp_enabled"] = False
    cfg["l2sp_lambda"] = 0.0
    return cfg


def _runs_for_split(split: str) -> list[dict]:
    split_key = str(split).strip().lower()
    if split_key not in RUNS_BY_SPLIT:
        raise ValueError(f" split={split!r}， {sorted(RUNS_BY_SPLIT)}")
    return [
        {"run_idx": idx, "label": label, "train_csv": train, "val_csv": val, "test_csv": test}
        for idx, (label, train, val, test) in enumerate(RUNS_BY_SPLIT[split_key], start=1)
    ]


def _find_run_config(runs: list[dict], run_idx: int) -> dict:
    for item in runs:
        if int(item["run_idx"]) == int(run_idx):
            return item
    raise ValueError(f" run_idx={run_idx}， {[x['run_idx'] for x in runs]}")


def _check_required_files(run_cfg: dict, cfg: dict) -> None:
    required_files = [
        run_cfg["train_csv"],
        run_cfg["val_csv"],
        run_cfg["test_csv"],
        cfg["more_weight_path"],
    ]
    missing = [p for p in required_files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "，：\n"
            + "\n".join([f"  - {p}" for p in missing])
        )


def _make_run_config(cfg: dict, run_idx: int) -> RunConfig:
    return RunConfig(
        seed=cfg["seed"],
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        min_epochs=cfg["min_epochs"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        l2sp_enabled=cfg["l2sp_enabled"],
        l2sp_lambda=cfg["l2sp_lambda"],
        l2sp_modules=cfg["l2sp_modules"],
        l2sp_exclude_bias_norm=cfg["l2sp_exclude_bias_norm"],
        num_workers=cfg["num_workers"],
        cache_graphs=cfg["cache_graphs"],
        drop_invalid_solvent_in_graph=cfg["drop_invalid_solvent_in_graph"],
        solute_encoder_type=cfg["solute_encoder_type"],
        more_weight_path=cfg["more_weight_path"],
        more_num_layer=cfg["more_num_layer"],
        more_emb_dim=cfg["more_emb_dim"],
        more_jk=cfg["more_jk"],
        more_dropout_ratio=cfg["more_dropout_ratio"],
        more_gnn_type=cfg["more_gnn_type"],
        more_pooling=cfg["more_pooling"],
        more_frozen=cfg["more_frozen"],
        more_unfrozen_layers=cfg["more_unfrozen_layers"],
        graph_transformer_hidden=cfg["graph_transformer_hidden"],
        graph_transformer_layers=cfg["graph_transformer_layers"],
        graph_transformer_out=cfg["graph_transformer_out"],
        graph_transformer_heads=cfg["graph_transformer_heads"],
        graph_transformer_edge_dim=cfg["graph_transformer_edge_dim"],
        graph_transformer_dropout=cfg["graph_transformer_dropout"],
        graph_transformer_use_edge_attr=cfg["graph_transformer_use_edge_attr"],
        graph_transformer_beta=cfg["graph_transformer_beta"],
        graph_transformer_readout=cfg["graph_transformer_readout"],
        graph_transformer_readout_heads=cfg["graph_transformer_readout_heads"],
        graph_transformer_readout_layers=cfg["graph_transformer_readout_layers"],
        graph_transformer_readout_ffn_dim=cfg["graph_transformer_readout_ffn_dim"],
        graph_transformer_readout_dropout=cfg["graph_transformer_readout_dropout"],
        solvent_mode=cfg["solvent_mode"],
        morgan_radius=cfg["morgan_radius"],
        morgan_n_bits=cfg["morgan_n_bits"],
        morgan_use_chirality=cfg["morgan_use_chirality"],
        solvent_graph_encoder_type=cfg["solvent_graph_encoder_type"],
        solvent_graph_transformer_hidden=cfg["solvent_graph_transformer_hidden"],
        solvent_graph_transformer_layers=cfg["solvent_graph_transformer_layers"],
        solvent_graph_transformer_out=cfg["solvent_graph_transformer_out"],
        solvent_graph_transformer_heads=cfg["solvent_graph_transformer_heads"],
        solvent_graph_transformer_edge_dim=cfg["solvent_graph_transformer_edge_dim"],
        solvent_graph_transformer_dropout=cfg["solvent_graph_transformer_dropout"],
        solvent_graph_transformer_use_edge_attr=cfg["solvent_graph_transformer_use_edge_attr"],
        solvent_graph_transformer_beta=cfg["solvent_graph_transformer_beta"],
        fusion_type=cfg["fusion_type"],
        fusion_attention_dim=cfg["fusion_attention_dim"],
        fusion_attention_heads=cfg["fusion_attention_heads"],
        fusion_attention_layers=cfg["fusion_attention_layers"],
        fusion_attention_ffn_dim=cfg["fusion_attention_ffn_dim"],
        fusion_attention_dropout=cfg["fusion_attention_dropout"],
        fusion_output_mode=cfg["fusion_output_mode"],
        mlp_hidden=cfg["mlp_hidden"],
        mlp_layers=cfg["mlp_layers"],
        mlp_dropout=cfg["mlp_dropout"],
        target_names=("abs", "emi", "plqy", "em"),
        run_idx=run_idx,
        cv_fold=run_idx,
    )


def run_one(variant_config: dict, split: str, run_idx: int) -> None:
    cfg = _merged_config(variant_config)
    _patch_more_only_encoder_if_needed(cfg["solute_encoder_type"])

    runs = _runs_for_split(split)
    run_cfg = _find_run_config(runs, run_idx)
    _check_required_files(run_cfg, cfg)

    output_root = str(Path(cfg["output_root"]) / run_cfg["label"])
    run_config = _make_run_config(cfg, run_idx=run_idx)
    io_cfg = RunIOConfig(
        train_csv=run_cfg["train_csv"],
        val_csv=run_cfg["val_csv"],
        test_csv=run_cfg["test_csv"],
        output_root=output_root,
        early_stopping_patience=cfg["early_stopping_patience"],
    )

    print("=" * 80)
    print(f"[HPO] variant          = {cfg['output_root']}")
    print(f"[HPO] split            = {split}")
    print(f"[Run {run_idx}] label   = {run_cfg['label']}")
    print(f"train_csv              = {run_cfg['train_csv']}")
    print(f"val_csv                = {run_cfg['val_csv']}")
    print(f"test_csv               = {run_cfg['test_csv']}")
    print(f"solute_encoder         = {cfg['solute_encoder_type']}")
    print(f"MORE frozen            = {cfg['more_frozen']} | unfrozen_layers={cfg['more_unfrozen_layers']}")
    print(f"L2SP enabled           = {cfg['l2sp_enabled']}")
    print(
        "solute GT              = "
        f"h{cfg['graph_transformer_hidden']} x L{cfg['graph_transformer_layers']} "
        f"out{cfg['graph_transformer_out']} heads{cfg['graph_transformer_heads']} "
        f"drop{cfg['graph_transformer_dropout']} readout={cfg['graph_transformer_readout']}"
    )
    print(
        "solvent GT             = "
        f"h{cfg['solvent_graph_transformer_hidden']} x L{cfg['solvent_graph_transformer_layers']} "
        f"out{cfg['solvent_graph_transformer_out']} heads{cfg['solvent_graph_transformer_heads']} "
        f"drop{cfg['solvent_graph_transformer_dropout']}"
    )
    print(
        "fusion SA              = "
        f"dim{cfg['fusion_attention_dim']} heads{cfg['fusion_attention_heads']} "
        f"layers{cfg['fusion_attention_layers']} ffn={cfg['fusion_attention_ffn_dim']} "
        f"drop{cfg['fusion_attention_dropout']} mode={cfg['fusion_output_mode']}"
    )
    print(
        "MLP                    = "
        f"h{cfg['mlp_hidden']} x L{cfg['mlp_layers']} drop{cfg['mlp_dropout']} | "
        f"lr={cfg['lr']} wd={cfg['weight_decay']} bs={cfg['batch_size']}"
    )
    print(f"output_root            = {output_root}")
    print("=" * 80)

    run_from_io_config(cfg=run_config, io_cfg=io_cfg)


def run_all_in_independent_processes(script_path: Path, split: str) -> None:
    for item in _runs_for_split(split):
        run_idx = int(item["run_idx"])
        cmd = [sys.executable, str(script_path), "--single_run", str(run_idx), "--split", split]

        print("\n" + "#" * 80)
        print(f"[Launcher] start {item['label']}")
        print("[Launcher] command:", " ".join(cmd))
        print("#" * 80)

        if os.name == "nt":
            proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            proc = subprocess.Popen(cmd)

        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"{item['label']} ，return_code={return_code}。"
                " run ， run 。"
            )

        print(f"[Launcher] finished {item['label']}")

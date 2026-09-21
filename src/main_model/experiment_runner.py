from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from data_loading import make_train_val_test_loaders
from experiment_utils import seed_everything
from model_factory import build_system
from trainer import train_experiment


@dataclass
class RunConfig:
    
    seed: int = 42
    batch_size: int = 256
    epochs: int = 5000
    min_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4

    
    
    
    l2sp_enabled: bool = False
    l2sp_lambda: float = 1e-4
    l2sp_modules: tuple[str, ...] = ("mol_encoder.more_encoder",)
    l2sp_exclude_bias_norm: bool = True

    
    num_workers: int = 4
    cache_graphs: bool = True
    drop_invalid_solvent_in_graph: bool = True

    
    
    solute_encoder_type: str = "graph_transformer"

    
    
    
    more_weight_path: Optional[str] = None
    more_num_layer: int = 5
    more_emb_dim: int = 300
    more_jk: str = "last"
    more_dropout_ratio: float = 0.0
    more_gnn_type: str = "gin"
    more_pooling: str = "mean"
    more_frozen: bool = True
    more_unfrozen_layers: Optional[int] = None

    
    morgan_radius: int = 4
    morgan_n_bits: int = 256
    morgan_use_chirality: bool = False

    
    rdkit_descriptor_names: Optional[tuple[str, ...]] = None
    rdkit_nan_value: float = 0.0
    rdkit_inf_value: float = 0.0

    
    graph_transformer_hidden: int = 128
    graph_transformer_layers: int = 3
    graph_transformer_out: int = 64
    graph_transformer_heads: int = 4
    graph_transformer_edge_dim: int = 6
    graph_transformer_dropout: float = 0.1
    graph_transformer_use_edge_attr: bool = True
    graph_transformer_beta: bool = False
    
    
    
    graph_transformer_readout: str = "attention"
    graph_transformer_readout_heads: Optional[int] = None
    graph_transformer_readout_layers: int = 1
    graph_transformer_readout_ffn_dim: Optional[int] = None
    graph_transformer_readout_dropout: Optional[float] = None

    
    solvent_graph_encoder_type: Optional[str] = None
    solvent_gat_hidden: int = 128
    solvent_gat_layers: int = 3
    solvent_gat_out: int = 64
    solvent_gat_heads: int = 4
    solvent_gat_concat: bool = True
    solvent_gat_dropout: float = 0.1
    solvent_graph_transformer_hidden: int = 128
    solvent_graph_transformer_layers: int = 3
    solvent_graph_transformer_out: int = 64
    solvent_graph_transformer_heads: int = 4
    solvent_graph_transformer_edge_dim: int = 6
    solvent_graph_transformer_dropout: float = 0.1
    solvent_graph_transformer_use_edge_attr: bool = True
    solvent_graph_transformer_beta: bool = False

    
    
    
    fusion_type: str = "concat"
    fusion_attention_dim: int = 128
    fusion_attention_heads: int = 4
    fusion_attention_layers: int = 1
    fusion_attention_ffn_dim: Optional[int] = None
    fusion_attention_dropout: float = 0.1
    fusion_output_mode: str = "cls"

    
    mlp_hidden: int = 512
    mlp_layers: int = 3
    mlp_dropout: float = 0.1

    
    early_stopping_min_delta: float = 1e-4
    early_stopping_mode: str = "max"
    early_stopping_verbose: bool = True

    
    lr_scheduler_enabled: bool = True
    lr_scheduler_name: str = "ReduceLROnPlateau"
    lr_scheduler_monitor: str = "val_loss"
    lr_scheduler_mode: str = "min"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 10
    lr_scheduler_threshold: float = 1e-4
    lr_scheduler_threshold_mode: str = "rel"
    lr_scheduler_cooldown: int = 0
    lr_scheduler_min_lr: float = 1e-6
    lr_scheduler_eps: float = 1e-8

    
    denom_eps: float = 1e-8
    solvent_mode: str = "MorganFP"
    target_names: tuple[str, ...] = ("abs", "emi", "plqy", "em")
    run_idx: int = 1
    cv_fold: Optional[int] = None
    full_train_eval_each_epoch: bool = False


DEFAULT_SUMMARY_CSV_NAME = "summary.csv"
DEFAULT_FINAL_TEST_CSV_NAME = "test_predictions.csv"
DEFAULT_BEST_MODEL_NAME = "best_model.pt"
DEFAULT_TRAIN_LOG_NAME = "training_log.npz"
DEFAULT_TRAIN_LOG_CSV_NAME = "training_log.csv"


@dataclass
class RunIOConfig:
    train_csv: str = "Train.csv"
    val_csv: str = "Val.csv"
    test_csv: str = "Test.csv"
    output_root: str = "./outputs"
    summary_csv_name: str = DEFAULT_SUMMARY_CSV_NAME
    final_test_csv_name: str = DEFAULT_FINAL_TEST_CSV_NAME
    best_model_name: str = DEFAULT_BEST_MODEL_NAME
    train_log_name: str = DEFAULT_TRAIN_LOG_NAME
    train_log_csv_name: str = DEFAULT_TRAIN_LOG_CSV_NAME
    early_stopping_patience: int = 200


def _build_namespace(
    *,
    cfg: RunConfig,
    io_cfg: RunIOConfig,
) -> SimpleNamespace:
    output_dir = Path(io_cfg.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    pin_memory = torch.cuda.is_available()
    persistent_workers = cfg.num_workers > 0
    prefetch_factor = 2 if cfg.num_workers > 0 else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return SimpleNamespace(
        TRAIN_CSV=str(io_cfg.train_csv),
        VAL_CSV=str(io_cfg.val_csv),
        TEST_CSV=str(io_cfg.test_csv),
        OUTPUT_DIR=str(output_dir),
        BEST_PATH=str(output_dir / io_cfg.best_model_name),
        LOG_PATH=str(output_dir / io_cfg.train_log_name),
        LOG_CSV_PATH=str(output_dir / io_cfg.train_log_csv_name),
        SEED=cfg.seed,
        BATCH_SIZE=cfg.batch_size,
        EPOCHS=cfg.epochs,
        MIN_EPOCHS=cfg.min_epochs,
        LR=cfg.lr,
        WEIGHT_DECAY=cfg.weight_decay,
        L2SP_ENABLED=cfg.l2sp_enabled,
        L2SP_LAMBDA=cfg.l2sp_lambda,
        L2SP_MODULES=cfg.l2sp_modules,
        L2SP_EXCLUDE_BIAS_NORM=cfg.l2sp_exclude_bias_norm,
        DENOM_EPS=cfg.denom_eps,
        NUM_WORKERS=cfg.num_workers,
        PIN_MEMORY=pin_memory,
        PERSISTENT_WORKERS=persistent_workers,
        PREFETCH_FACTOR=prefetch_factor,
        CACHE_GRAPHS=cfg.cache_graphs,
        DROP_INVALID_SOLVENT_IN_GRAPH=cfg.drop_invalid_solvent_in_graph,
        SOLUTE_ENCODER_TYPE=cfg.solute_encoder_type,
        MORE_WEIGHT_PATH=cfg.more_weight_path,
        MORE_NUM_LAYER=cfg.more_num_layer,
        MORE_EMB_DIM=cfg.more_emb_dim,
        MORE_JK=cfg.more_jk,
        MORE_DROPOUT_RATIO=cfg.more_dropout_ratio,
        MORE_GNN_TYPE=cfg.more_gnn_type,
        MORE_POOLING=cfg.more_pooling,
        MORE_FROZEN=cfg.more_frozen,
        MORE_UNFROZEN_LAYERS=cfg.more_unfrozen_layers,
        SOLVENT_MODE=cfg.solvent_mode,
        MORGAN_RADIUS=cfg.morgan_radius,
        MORGAN_N_BITS=cfg.morgan_n_bits,
        MORGAN_USE_CHIRALITY=cfg.morgan_use_chirality,
        RDKIT_DESCRIPTOR_NAMES=cfg.rdkit_descriptor_names,
        RDKIT_NAN_VALUE=cfg.rdkit_nan_value,
        RDKIT_INF_VALUE=cfg.rdkit_inf_value,
        TARGET_NAMES=list(cfg.target_names),
        GRAPH_TRANSFORMER_HIDDEN=cfg.graph_transformer_hidden,
        GRAPH_TRANSFORMER_LAYERS=cfg.graph_transformer_layers,
        GRAPH_TRANSFORMER_OUT=cfg.graph_transformer_out,
        GRAPH_TRANSFORMER_HEADS=cfg.graph_transformer_heads,
        GRAPH_TRANSFORMER_EDGE_DIM=cfg.graph_transformer_edge_dim,
        GRAPH_TRANSFORMER_DROPOUT=cfg.graph_transformer_dropout,
        GRAPH_TRANSFORMER_USE_EDGE_ATTR=cfg.graph_transformer_use_edge_attr,
        GRAPH_TRANSFORMER_BETA=cfg.graph_transformer_beta,
        GRAPH_TRANSFORMER_READOUT=cfg.graph_transformer_readout,
        GRAPH_TRANSFORMER_READOUT_HEADS=(
            cfg.graph_transformer_readout_heads
            if cfg.graph_transformer_readout_heads is not None
            else cfg.graph_transformer_heads
        ),
        GRAPH_TRANSFORMER_READOUT_LAYERS=cfg.graph_transformer_readout_layers,
        GRAPH_TRANSFORMER_READOUT_FFN_DIM=cfg.graph_transformer_readout_ffn_dim,
        GRAPH_TRANSFORMER_READOUT_DROPOUT=(
            cfg.graph_transformer_readout_dropout
            if cfg.graph_transformer_readout_dropout is not None
            else cfg.graph_transformer_dropout
        ),
        SOLVENT_GRAPH_ENCODER_TYPE=cfg.solvent_graph_encoder_type,
        SOLVENT_GAT_HIDDEN=cfg.solvent_gat_hidden,
        SOLVENT_GAT_LAYERS=cfg.solvent_gat_layers,
        SOLVENT_GAT_OUT=cfg.solvent_gat_out,
        SOLVENT_GAT_HEADS=cfg.solvent_gat_heads,
        SOLVENT_GAT_CONCAT=cfg.solvent_gat_concat,
        SOLVENT_GAT_DROPOUT=cfg.solvent_gat_dropout,
        SOLVENT_GRAPH_TRANSFORMER_HIDDEN=cfg.solvent_graph_transformer_hidden,
        SOLVENT_GRAPH_TRANSFORMER_LAYERS=cfg.solvent_graph_transformer_layers,
        SOLVENT_GRAPH_TRANSFORMER_OUT=cfg.solvent_graph_transformer_out,
        SOLVENT_GRAPH_TRANSFORMER_HEADS=cfg.solvent_graph_transformer_heads,
        SOLVENT_GRAPH_TRANSFORMER_EDGE_DIM=cfg.solvent_graph_transformer_edge_dim,
        SOLVENT_GRAPH_TRANSFORMER_DROPOUT=cfg.solvent_graph_transformer_dropout,
        SOLVENT_GRAPH_TRANSFORMER_USE_EDGE_ATTR=cfg.solvent_graph_transformer_use_edge_attr,
        SOLVENT_GRAPH_TRANSFORMER_BETA=cfg.solvent_graph_transformer_beta,
        FUSION_TYPE=cfg.fusion_type,
        FUSION_ATTENTION_DIM=cfg.fusion_attention_dim,
        FUSION_ATTENTION_HEADS=cfg.fusion_attention_heads,
        FUSION_ATTENTION_LAYERS=cfg.fusion_attention_layers,
        FUSION_ATTENTION_FFN_DIM=cfg.fusion_attention_ffn_dim,
        FUSION_ATTENTION_DROPOUT=cfg.fusion_attention_dropout,
        FUSION_OUTPUT_MODE=cfg.fusion_output_mode,
        MLP_HIDDEN=cfg.mlp_hidden,
        MLP_LAYERS=cfg.mlp_layers,
        MLP_DROPOUT=cfg.mlp_dropout,
        EARLY_STOPPING_PATIENCE=int(io_cfg.early_stopping_patience),
        EARLY_STOPPING_MIN_DELTA=cfg.early_stopping_min_delta,
        EARLY_STOPPING_MODE=cfg.early_stopping_mode,
        EARLY_STOPPING_VERBOSE=cfg.early_stopping_verbose,
        LR_SCHEDULER_ENABLED=cfg.lr_scheduler_enabled,
        LR_SCHEDULER_NAME=cfg.lr_scheduler_name,
        LR_SCHEDULER_MONITOR=cfg.lr_scheduler_monitor,
        LR_SCHEDULER_MODE=cfg.lr_scheduler_mode,
        LR_SCHEDULER_FACTOR=cfg.lr_scheduler_factor,
        LR_SCHEDULER_PATIENCE=cfg.lr_scheduler_patience,
        LR_SCHEDULER_THRESHOLD=cfg.lr_scheduler_threshold,
        LR_SCHEDULER_THRESHOLD_MODE=cfg.lr_scheduler_threshold_mode,
        LR_SCHEDULER_COOLDOWN=cfg.lr_scheduler_cooldown,
        LR_SCHEDULER_MIN_LR=cfg.lr_scheduler_min_lr,
        LR_SCHEDULER_EPS=cfg.lr_scheduler_eps,
        FULL_TRAIN_EVAL_EACH_EPOCH=cfg.full_train_eval_each_epoch,
        DEVICE=device,
        RUN_IDX=cfg.run_idx,
        CV_FOLD=cfg.cv_fold,
    )


def _safe_scalar_from_npz(obj, key: str) -> float:
    if key not in obj:
        return float("nan")
    arr = np.asarray(obj[key], dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def _load_best_val_r2_from_checkpoint(best_path: str) -> float:
    path = Path(best_path)
    if not path.exists():
        return float("nan")
    try:
        ckpt = torch.load(path, map_location="cpu")
        return float(ckpt.get("val_r2", float("nan")))
    except Exception:
        return float("nan")


def collect_run_metrics(cfg_ns: SimpleNamespace) -> dict:
    row = {
        "run_idx": int(cfg_ns.RUN_IDX),
        "train_csv": cfg_ns.TRAIN_CSV,
        "val_csv": cfg_ns.VAL_CSV,
        "test_csv": cfg_ns.TEST_CSV,
        "best_epoch": float("nan"),
        "best_val_r2": float("nan"),
        "train_solute_skipped_rows": float("nan"),
        "train_solvent_skipped_rows": float("nan"),
        "val_solute_skipped_rows": float("nan"),
        "val_solvent_skipped_rows": float("nan"),
        "test_solute_skipped_rows": float("nan"),
        "test_solvent_skipped_rows": float("nan"),
        "log_path": cfg_ns.LOG_PATH,
        "log_csv_path": cfg_ns.LOG_CSV_PATH,
        "best_path": cfg_ns.BEST_PATH,
    }

    log_path = Path(cfg_ns.LOG_PATH)
    if not log_path.exists():
        return row

    with np.load(log_path) as obj:
        row["best_epoch"] = _safe_scalar_from_npz(obj, "best_epoch")
        row["best_val_r2"] = _load_best_val_r2_from_checkpoint(cfg_ns.BEST_PATH)
        if not np.isfinite(row["best_val_r2"]):
            val_r2 = np.asarray(obj["val_r2"], dtype=float).reshape(-1) if "val_r2" in obj else np.array([], dtype=float)
            best_epoch = row["best_epoch"]
            if np.isfinite(best_epoch):
                best_epoch = int(best_epoch)
                if 1 <= best_epoch <= val_r2.size:
                    row["best_val_r2"] = float(val_r2[best_epoch - 1])

        row["train_solute_skipped_rows"] = _safe_scalar_from_npz(obj, "train_solute_skipped_rows")
        row["train_solvent_skipped_rows"] = _safe_scalar_from_npz(obj, "train_solvent_skipped_rows")
        row["val_solute_skipped_rows"] = _safe_scalar_from_npz(obj, "val_solute_skipped_rows")
        row["val_solvent_skipped_rows"] = _safe_scalar_from_npz(obj, "val_solvent_skipped_rows")
        row["test_solute_skipped_rows"] = _safe_scalar_from_npz(obj, "test_solute_skipped_rows")
        row["test_solvent_skipped_rows"] = _safe_scalar_from_npz(obj, "test_solvent_skipped_rows")
    return row


def save_summary(output_root: str, summary_csv_name: str, rows: list[dict]) -> None:
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / summary_csv_name
    header = [
        "run_idx",
        "train_csv",
        "val_csv",
        "test_csv",
        "best_epoch",
        "best_val_r2",
        "train_solute_skipped_rows",
        "train_solvent_skipped_rows",
        "val_solute_skipped_rows",
        "val_solvent_skipped_rows",
        "test_solute_skipped_rows",
        "test_solvent_skipped_rows",
        "log_path",
        "log_csv_path",
        "best_path",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Summary] saved -> {path}")


def save_final_test(output_root: str, final_test_csv_name: str, rows: list[dict]) -> None:
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / final_test_csv_name
    header = [
        "run_idx",
        "train_csv",
        "val_csv",
        "test_csv",
        "best_epoch",
        "best_val_r2",
        "final_test_loss",
        "final_test_mae",
        "final_test_rmse",
        "final_test_r2",
        "final_test_abs_loss",
        "final_test_emi_loss",
        "final_test_plqy_loss",
        "final_test_em_loss",
        "final_test_abs_mae",
        "final_test_emi_mae",
        "final_test_plqy_mae",
        "final_test_em_mae",
        "final_test_abs_rmse",
        "final_test_emi_rmse",
        "final_test_plqy_rmse",
        "final_test_em_rmse",
        "final_test_abs_r2",
        "final_test_emi_r2",
        "final_test_plqy_r2",
        "final_test_em_r2",
        "best_path",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[FinalTest] saved -> {path}")


def run_and_save(
    *,
    cfg: RunConfig,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    output_root: str,
    summary_csv_name: str = DEFAULT_SUMMARY_CSV_NAME,
    final_test_csv_name: str = DEFAULT_FINAL_TEST_CSV_NAME,
    best_model_name: str = DEFAULT_BEST_MODEL_NAME,
    train_log_name: str = DEFAULT_TRAIN_LOG_NAME,
    train_log_csv_name: str = DEFAULT_TRAIN_LOG_CSV_NAME,
    early_stopping_patience: int = 200,
) -> None:
    io_cfg = RunIOConfig(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        output_root=output_root,
        summary_csv_name=summary_csv_name,
        final_test_csv_name=final_test_csv_name,
        best_model_name=best_model_name,
        train_log_name=train_log_name,
        train_log_csv_name=train_log_csv_name,
        early_stopping_patience=early_stopping_patience,
    )
    cfg_ns = _build_namespace(cfg=cfg, io_cfg=io_cfg)

    seed_everything(cfg_ns.SEED)

    train_loader, val_loader, test_loader, _ = make_train_val_test_loaders(
        train_csv_path=cfg_ns.TRAIN_CSV,
        val_csv_path=cfg_ns.VAL_CSV,
        test_csv_path=cfg_ns.TEST_CSV,
        solvent_mode=cfg_ns.SOLVENT_MODE,
        batch_size=cfg_ns.BATCH_SIZE,
        seed=cfg_ns.SEED,
        num_workers=cfg_ns.NUM_WORKERS,
        pin_memory=cfg_ns.PIN_MEMORY,
        persistent_workers=cfg_ns.PERSISTENT_WORKERS,
        prefetch_factor=cfg_ns.PREFETCH_FACTOR,
        cache_graphs=cfg_ns.CACHE_GRAPHS,
        drop_invalid_solvent_in_graph=cfg_ns.DROP_INVALID_SOLVENT_IN_GRAPH,
        morgan_kwargs={
            "radius": cfg_ns.MORGAN_RADIUS,
            "n_bits": cfg_ns.MORGAN_N_BITS,
            "use_chirality": cfg_ns.MORGAN_USE_CHIRALITY,
        },
        rdkit_kwargs={
            "descriptor_names": cfg_ns.RDKIT_DESCRIPTOR_NAMES,
            "nan_value": cfg_ns.RDKIT_NAN_VALUE,
            "inf_value": cfg_ns.RDKIT_INF_VALUE,
        },
        cv_fold=cfg_ns.CV_FOLD,
    )

    print("\n" + "=" * 80)
    print(f"[Device] using device = {cfg_ns.DEVICE}")
    print(f"[Input ] train={cfg_ns.TRAIN_CSV}")
    print(f"[Input ] val  ={cfg_ns.VAL_CSV}")
    print(f"[Input ] test ={cfg_ns.TEST_CSV}")
    print(f"[Solute] encoder={cfg_ns.SOLUTE_ENCODER_TYPE}")
    solute_encoder_key = str(cfg_ns.SOLUTE_ENCODER_TYPE).strip().lower()
    if "more_online" in solute_encoder_key or "online_more" in solute_encoder_key:
        print(f"[Solute] MORE_weight_path={cfg_ns.MORE_WEIGHT_PATH}")
        print(
            f"[Solute] MORE frozen={cfg_ns.MORE_FROZEN} | "
            f"unfrozen_layers={cfg_ns.MORE_UNFROZEN_LAYERS} | pooling={cfg_ns.MORE_POOLING}"
        )
    print(
        f"[L2SP  ] enabled={cfg_ns.L2SP_ENABLED} | lambda={cfg_ns.L2SP_LAMBDA} | "
        f"modules={cfg_ns.L2SP_MODULES} | exclude_bias_norm={cfg_ns.L2SP_EXCLUDE_BIAS_NORM}"
    )
    print(
        f"[Solvent] mode={cfg_ns.SOLVENT_MODE} | drop_invalid_solvent={cfg_ns.DROP_INVALID_SOLVENT_IN_GRAPH} | "
        f"radius={cfg_ns.MORGAN_RADIUS} | n_bits={cfg_ns.MORGAN_N_BITS} | use_chirality={cfg_ns.MORGAN_USE_CHIRALITY}"
    )
    print(f"[Output] dir  ={cfg_ns.OUTPUT_DIR}")
    print(f"[Output] ckpt ={cfg_ns.BEST_PATH}")
    print(f"[Output] npz  ={cfg_ns.LOG_PATH}")
    print(f"[Output] csv  ={cfg_ns.LOG_CSV_PATH}")
    print(f"[EarlyStopping] patience={cfg_ns.EARLY_STOPPING_PATIENCE}")
    print("=" * 80)

    sample_batch = next(iter(train_loader))
    system = build_system(sample_batch=sample_batch, cfg=cfg_ns)

    final_test_row = train_experiment(
        system=system,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg_ns
    )

    summary_row = collect_run_metrics(cfg_ns)
    summary_rows = [summary_row]
    save_summary(output_root, summary_csv_name, summary_rows)

    final_test_row = dict(final_test_row)
    final_test_row.setdefault("run_idx", int(cfg_ns.RUN_IDX))
    final_test_row.setdefault("train_csv", cfg_ns.TRAIN_CSV)
    final_test_row.setdefault("val_csv", cfg_ns.VAL_CSV)
    final_test_row.setdefault("test_csv", cfg_ns.TEST_CSV)
    final_test_row.setdefault("best_epoch", summary_row.get("best_epoch", float("nan")))
    final_test_row.setdefault("best_val_r2", summary_row.get("best_val_r2", float("nan")))
    final_test_row.setdefault("best_path", cfg_ns.BEST_PATH)
    save_final_test(output_root, final_test_csv_name, [final_test_row])


def run_from_io_config(*, cfg: RunConfig, io_cfg: RunIOConfig) -> None:
    run_and_save(
        cfg=cfg,
        train_csv=io_cfg.train_csv,
        val_csv=io_cfg.val_csv,
        test_csv=io_cfg.test_csv,
        output_root=io_cfg.output_root,
        summary_csv_name=io_cfg.summary_csv_name,
        final_test_csv_name=io_cfg.final_test_csv_name,
        best_model_name=io_cfg.best_model_name,
        train_log_name=io_cfg.train_log_name,
        train_log_csv_name=io_cfg.train_log_csv_name,
        early_stopping_patience=io_cfg.early_stopping_patience,
    )

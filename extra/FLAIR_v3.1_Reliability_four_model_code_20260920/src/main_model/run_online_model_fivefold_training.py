from __future__ import annotations

import argparse
from pathlib import Path
import sys

from run_model_training_workflow import RUNS_BY_SPLIT, run_all_in_independent_processes, run_one


VARIANT_CONFIG = {
    "output_root": "../../models/main_model",
    "batch_size": 256,
    "lr": 4e-4,
    "weight_decay": 2e-4,
    "epochs": 2000,
    "min_epochs": 70,
    "early_stopping_patience": 30,
    "solute_encoder_type": "graph_transformer_more_online_ln_adapter_gt_concat",
    "more_weight_path": "../../models/pretrained/MORE.pth",
    "more_num_layer": 5,
    "more_emb_dim": 300,
    "more_jk": "last",
    "more_dropout_ratio": 0.0,
    "more_gnn_type": "gin",
    "more_pooling": "mean",
    "more_frozen": True,
    "more_unfrozen_layers": 0,
    "l2sp_enabled": False,
    "l2sp_lambda": 0.0,
    "num_workers": 4,
    "cache_graphs": True,
    "drop_invalid_solvent_in_graph": True,
    "solvent_mode": "morgan",
    "morgan_radius": 4,
    "morgan_n_bits": 256,
    "morgan_use_chirality": False,
    "graph_transformer_hidden": 288,
    "graph_transformer_layers": 4,
    "graph_transformer_out": 288,
    "graph_transformer_heads": 8,
    "graph_transformer_dropout": 0.10,
    "graph_transformer_readout": "attention",
    "graph_transformer_readout_heads": 8,
    "graph_transformer_readout_layers": 1,
    "graph_transformer_readout_ffn_dim": None,
    "graph_transformer_readout_dropout": 0.10,
    "solvent_graph_encoder_type": "transformer",
    "solvent_graph_transformer_hidden": 288,
    "solvent_graph_transformer_layers": 3,
    "solvent_graph_transformer_out": 288,
    "solvent_graph_transformer_heads": 8,
    "solvent_graph_transformer_dropout": 0.10,
    "fusion_type": "self_attention",
    "fusion_attention_dim": 256,
    "fusion_attention_heads": 8,
    "fusion_attention_layers": 2,
    "fusion_attention_ffn_dim": 1024,
    "fusion_attention_dropout": 0.18,
    "fusion_output_mode": "cls",
    "mlp_hidden": 640,
    "mlp_layers": 3,
    "mlp_dropout": 0.30,
}


RUNS_BY_SPLIT["ae"] = [
    (
        f"fold_{idx:02d}",
        "../../data/splits/deployment/deployment.csv",
        "../../data/splits/deployment/deployment.csv",
        "../../data/splits/deployment/deployment_test.csv",
    )
    for idx in range(1, 6)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the deployment 5-fold DL main models.")
    parser.add_argument("--single_run", type=int, default=None, help="Run only one fold, e.g. --single_run 1.")
    parser.add_argument("--split", choices=["ae"], default="ae", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.single_run is not None:
        run_one(variant_config=VARIANT_CONFIG, split="ae", run_idx=int(args.single_run))
    else:
        run_all_in_independent_processes(Path(sys.argv[0]).name, split="ae")

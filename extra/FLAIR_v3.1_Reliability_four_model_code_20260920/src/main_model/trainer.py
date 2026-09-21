from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from MLP import eval_step, masked_mse_loss_with_stats, r2_ignore_masked, train_step
from log_utils import save_training_log
from model_factory import ensure_same_device, get_module_device


TARGET_DIM = 4


def move_batch_to_device(batch: Any, device: torch.device, non_blocking: bool = True):
    if hasattr(batch, "to"):
        return batch.to(device, non_blocking=non_blocking)
    return batch


def get_module_dtype(module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def safe_mean(values: list[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(sum(values) / len(values))


def safe_mean_vector(values: list[np.ndarray], dim: int = 4) -> np.ndarray:
    if len(values) == 0:
        return np.full((dim,), np.nan, dtype=float)

    arr = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(arr, axis=0)
    return out.astype(float)


def _get_solvent_feat(batch) -> torch.Tensor:
    if hasattr(batch, "solvent_feat"):
        feat = batch.solvent_feat
    elif hasattr(batch, "solv_cond"):
        feat = batch.solv_cond
    else:
        raise ValueError("batch  solvent_feat / solv_cond。 data_loading.py 。")

    if not isinstance(feat, torch.Tensor):
        feat = torch.as_tensor(feat)

    if feat.ndim == 1:
        feat = feat.view(1, -1)
    elif feat.ndim == 3 and feat.size(1) == 1:
        feat = feat.view(feat.size(0), feat.size(-1))
    elif feat.ndim != 2:
        raise ValueError(f"solvent_feat  [B,D]  [B,1,D]， {tuple(feat.shape)}")

    return feat.float()


def _get_solvent_graph_output(
    batch,
    system: SimpleNamespace,
    *,
    return_node_embeddings: bool = False,
):
    if not hasattr(batch, "solvent_x"):
        raise ValueError("batch  solvent_x。 data_loading.py 。")
    if not hasattr(batch, "solvent_edge_index"):
        raise ValueError("batch  solvent_edge_index。 data_loading.py 。")

    work_device = ensure_same_device(system.mol_encoder, system.mlp, system.sol_encoder)
    if getattr(system, "use_fusion_self_attention", False):
        work_device = ensure_same_device(system.mol_encoder, system.mlp, system.sol_encoder, system.fusion_encoder)
    work_dtype = get_module_dtype(system.mlp)

    sol_x = batch.solvent_x.to(device=work_device, dtype=work_dtype, non_blocking=True)
    sol_edge_index = batch.solvent_edge_index.to(device=work_device, non_blocking=True)

    if hasattr(batch, "solvent_x_batch"):
        sol_batch = batch.solvent_x_batch.to(device=work_device, non_blocking=True)
    else:
        sol_batch = torch.zeros(sol_x.size(0), dtype=torch.long, device=work_device)

    use_edge_attr = bool(getattr(system, "use_solvent_edge_attr", False))
    if use_edge_attr and hasattr(batch, "solvent_edge_attr"):
        sol_edge_attr = batch.solvent_edge_attr.to(device=work_device, dtype=work_dtype, non_blocking=True)
    else:
        sol_edge_attr = None

    if return_node_embeddings:
        sol_vec, solvent_node_tokens = system.sol_encoder(
            x=sol_x,
            edge_index=sol_edge_index,
            batch=sol_batch,
            edge_attr=sol_edge_attr,
            return_node_embeddings=True,
        )
        return sol_vec, solvent_node_tokens, sol_batch

    sol_vec = system.sol_encoder(
        x=sol_x,
        edge_index=sol_edge_index,
        batch=sol_batch,
        edge_attr=sol_edge_attr,
    )

    return sol_vec


def _get_solvent_graph_vec(batch, system: SimpleNamespace) -> torch.Tensor:
    return _get_solvent_graph_output(batch, system, return_node_embeddings=False)


def build_x_vec(batch, system: SimpleNamespace) -> torch.Tensor:
    modules_for_device = [system.mol_encoder, system.mlp]
    if getattr(system, "use_solvent_graph_encoder", False):
        modules_for_device.append(system.sol_encoder)
    if getattr(system, "use_fusion_self_attention", False):
        modules_for_device.append(system.fusion_encoder)

    work_device = ensure_same_device(*modules_for_device)
    work_dtype = get_module_dtype(system.mlp)

    use_fusion = bool(getattr(system, "use_fusion_self_attention", False))

    if use_fusion:
        
        mol_vec, solute_node_tokens = system.mol_encoder(data=batch, return_node_embeddings=True)
        if solute_node_tokens is None:
            raise ValueError(
                " mol_encoder  solute_node_tokens，"
                " token-level Self-Attention。"
            )

        if not getattr(system, "use_solvent_graph_encoder", False):
            raise ValueError(
                "token-level Self-Attention  solvent graph encoder，"
                " solvent_graph_encoder_type='transformer'。"
            )

        sol_vec, solvent_node_tokens, solvent_batch = _get_solvent_graph_output(
            batch,
            system,
            return_node_embeddings=True,
        )

        if hasattr(batch, "batch") and batch.batch is not None:
            solute_batch = batch.batch.to(device=work_device, non_blocking=True)
        else:
            solute_batch = torch.zeros(
                solute_node_tokens.size(0),
                dtype=torch.long,
                device=work_device,
            )
    else:
        mol_vec = system.mol_encoder(data=batch)
        solute_node_tokens = None
        solute_batch = None
        solvent_node_tokens = None
        solvent_batch = None

        if getattr(system, "use_solvent_graph_encoder", False):
            sol_vec = _get_solvent_graph_vec(batch, system)
        else:
            sol_vec = _get_solvent_feat(batch).to(device=work_device, dtype=work_dtype, non_blocking=True)

    if mol_vec.device != work_device or mol_vec.dtype != work_dtype:
        mol_vec = mol_vec.to(device=work_device, dtype=work_dtype, non_blocking=True)

    if sol_vec.device != work_device or sol_vec.dtype != work_dtype:
        sol_vec = sol_vec.to(device=work_device, dtype=work_dtype, non_blocking=True)

    if use_fusion:
        solute_node_tokens = solute_node_tokens.to(device=work_device, dtype=work_dtype, non_blocking=True)
        solvent_node_tokens = solvent_node_tokens.to(device=work_device, dtype=work_dtype, non_blocking=True)
        solute_batch = solute_batch.to(device=work_device, non_blocking=True)
        solvent_batch = solvent_batch.to(device=work_device, non_blocking=True)

        interaction_vec = system.fusion_encoder(
            solute_tokens=solute_node_tokens,
            solute_batch=solute_batch,
            solvent_tokens=solvent_node_tokens,
            solvent_batch=solvent_batch,
        )

        if interaction_vec.device != work_device or interaction_vec.dtype != work_dtype:
            interaction_vec = interaction_vec.to(device=work_device, dtype=work_dtype, non_blocking=True)

        
        x_vec = torch.cat([mol_vec, sol_vec, interaction_vec], dim=1)
    else:
        
        x_vec = torch.cat([mol_vec, sol_vec], dim=1)

    if x_vec.ndim != 2 or x_vec.size(1) != system.mlp_input_dim:
        raise ValueError(
            f"build_x_vec ： [B, {system.mlp_input_dim}]， {tuple(x_vec.shape)}"
        )

    return x_vec


def _to_2d_cpu_tensor(x) -> torch.Tensor | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        out = x.detach().cpu()
    else:
        out = torch.as_tensor(x)
    if out.ndim == 3 and out.size(1) == 1:
        out = out.view(out.size(0), out.size(-1))
    return out.float()


def _get_target_scale_and_bias(loader) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    dataset = getattr(loader, "dataset", None)
    target_std = getattr(dataset, "target_std", None)
    target_mean = getattr(dataset, "target_mean", None)

    std_tensor = None if target_std is None else torch.as_tensor(target_std, dtype=torch.float32).view(1, -1)
    mean_tensor = None if target_mean is None else torch.as_tensor(target_mean, dtype=torch.float32).view(1, -1)
    return std_tensor, mean_tensor


def _as_tuple_config(value) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return tuple(x.strip() for x in value.replace(",", "+").split("+") if x.strip())
    return tuple(str(x).strip() for x in value if str(x).strip())


def _resolve_attr_path(root, path: str):
    obj = root
    for part in str(path).split("."):
        if not part:
            continue
        obj = getattr(obj, part)
    return obj


def _l2sp_excluded_param(name: str) -> bool:
    key = str(name).lower()
    if key.endswith(".bias") or key == "bias":
        return True
    return any(token in key for token in ("norm", "bn", "layernorm", "batchnorm"))


def initialize_l2sp_regularizer(system: SimpleNamespace, cfg) -> None:
    enabled = bool(getattr(cfg, "L2SP_ENABLED", False))
    system.l2sp_refs = []

    if not enabled:
        system.l2sp_enabled = False
        return

    modules = _as_tuple_config(getattr(cfg, "L2SP_MODULES", ()))
    exclude_bias_norm = bool(getattr(cfg, "L2SP_EXCLUDE_BIAS_NORM", True))

    refs = []
    missing_modules = []
    for module_path in modules:
        try:
            module = _resolve_attr_path(system, module_path)
        except AttributeError:
            missing_modules.append(module_path)
            continue

        if not hasattr(module, "named_parameters"):
            missing_modules.append(module_path)
            continue

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if exclude_bias_norm and _l2sp_excluded_param(name):
                continue
            refs.append((f"{module_path}.{name}", param, param.detach().clone()))

    system.l2sp_enabled = len(refs) > 0
    system.l2sp_refs = refs

    if missing_modules:
        print(f"[L2SP] skipped missing modules: {', '.join(missing_modules)}")
    if len(refs) == 0:
        print(
            "[L2SP] enabled=True, but no trainable target parameters were found. "
            "If you want L2-SP on MORE.pth, set more_frozen=False."
        )
    else:
        n_params = sum(int(param.numel()) for _, param, _ in refs)
        print(
            f"[L2SP] enabled | lambda={float(getattr(cfg, 'L2SP_LAMBDA', 0.0)):.6g} | "
            f"modules={modules} | tensors={len(refs)} | params={n_params}"
        )


def compute_l2sp_loss(system: SimpleNamespace, cfg) -> torch.Tensor | None:
    refs = getattr(system, "l2sp_refs", None)
    if not refs:
        return None

    l2sp_lambda = float(getattr(cfg, "L2SP_LAMBDA", 0.0))
    if l2sp_lambda <= 0:
        return None

    penalty = None
    for _, param, ref in refs:
        ref = ref.to(device=param.device, dtype=param.dtype)
        value = (param - ref).pow(2).sum()
        penalty = value if penalty is None else penalty + value

    if penalty is None:
        return None
    return 0.5 * l2sp_lambda * penalty


def _inverse_standardize(
    y: torch.Tensor,
    target_std: torch.Tensor | None,
    target_mean: torch.Tensor | None,
) -> torch.Tensor:
    out = y.float()
    if target_std is not None:
        out = out * target_std.to(device=out.device, dtype=out.dtype)
    if target_mean is not None:
        out = out + target_mean.to(device=out.device, dtype=out.dtype)
    return out


def _masked_mae_with_stats(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor | None = None,
    per_task_average: bool = False,
):
    y_pred = y_pred.float()
    y_true = y_true.float()

    if y_mask is None:
        eff_mask = torch.isfinite(y_true).float()
    else:
        eff_mask = y_mask.float() * torch.isfinite(y_true).float()

    y_safe = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    abs_err = (y_pred - y_safe).abs()

    mae_task_list = []
    for j in range(y_true.size(1)):
        m = eff_mask[:, j] > 0.5
        if m.any():
            mae_task_list.append(abs_err[m, j].mean())
        else:
            mae_task_list.append(torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype))

    mae_task = torch.stack(mae_task_list)

    if per_task_average:
        valid = ~torch.isnan(mae_task)
        if valid.any():
            mae = mae_task[valid].mean()
        else:
            mae = torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype)
    else:
        denom = eff_mask.sum().clamp(min=1.0)
        mae = (abs_err * eff_mask).sum() / denom

    return mae, mae_task, abs_err, eff_mask


def _masked_rmse_with_stats(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor | None = None,
    per_task_average: bool = False,
):
    y_pred = y_pred.float()
    y_true = y_true.float()

    if y_mask is None:
        eff_mask = torch.isfinite(y_true).float()
    else:
        eff_mask = y_mask.float() * torch.isfinite(y_true).float()

    y_safe = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    sq_err = (y_pred - y_safe).pow(2)

    rmse_task_list = []
    for j in range(y_true.size(1)):
        m = eff_mask[:, j] > 0.5
        if m.any():
            rmse_task_list.append(torch.sqrt(sq_err[m, j].mean()))
        else:
            rmse_task_list.append(torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype))

    rmse_task = torch.stack(rmse_task_list)

    if per_task_average:
        valid = ~torch.isnan(rmse_task)
        if valid.any():
            rmse = rmse_task[valid].mean()
        else:
            rmse = torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype)
    else:
        denom = eff_mask.sum().clamp(min=1.0)
        rmse = torch.sqrt((sq_err * eff_mask).sum() / denom)

    return rmse, rmse_task, sq_err, eff_mask


def _compute_full_loader_metrics(
    pred_list,
    true_list,
    mask_list,
    *,
    target_std: torch.Tensor | None = None,
    target_mean: torch.Tensor | None = None,
):
    if len(pred_list) == 0:
        nan4 = np.full((TARGET_DIM,), np.nan, dtype=float)
        return {
            "loss": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "loss_task": nan4.copy(),
            "mae_task": nan4.copy(),
            "rmse_task": nan4.copy(),
            "r2_task": nan4.copy(),
        }

    y_pred_all = torch.cat(pred_list, dim=0).float()
    y_true_all = torch.cat(true_list, dim=0).float()

    if len(mask_list) == 0 or any(m is None for m in mask_list):
        y_mask_all = None
    else:
        y_mask_all = torch.cat(mask_list, dim=0).float()

    loss, loss_task, _, _ = masked_mse_loss_with_stats(
        y_pred=y_pred_all,
        y_true=y_true_all,
        y_mask=y_mask_all,
        per_task_average=False,
    )

    y_pred_abs = _inverse_standardize(y_pred_all, target_std=target_std, target_mean=target_mean)
    y_true_abs = _inverse_standardize(y_true_all, target_std=target_std, target_mean=target_mean)

    mae, mae_task, _, _ = _masked_mae_with_stats(
        y_pred=y_pred_abs,
        y_true=y_true_abs,
        y_mask=y_mask_all,
        per_task_average=False,
    )
    rmse, rmse_task, _, _ = _masked_rmse_with_stats(
        y_pred=y_pred_abs,
        y_true=y_true_abs,
        y_mask=y_mask_all,
        per_task_average=False,
    )
    r2_mean, r2_task, _ = r2_ignore_masked(
        y_pred=y_pred_all,
        y_true=y_true_all,
        y_mask=y_mask_all,
        require_nonzero_denom=False,
    )

    return {
        "loss": float(loss.item()) if not torch.isnan(loss) else float("nan"),
        "mae": float(mae.item()) if not torch.isnan(mae) else float("nan"),
        "rmse": float(rmse.item()) if not torch.isnan(rmse) else float("nan"),
        "r2": float(r2_mean.item()) if not torch.isnan(r2_mean) else float("nan"),
        "loss_task": loss_task.detach().cpu().numpy().astype(float),
        "mae_task": mae_task.detach().cpu().numpy().astype(float),
        "rmse_task": rmse_task.detach().cpu().numpy().astype(float),
        "r2_task": r2_task.detach().cpu().numpy().astype(float),
    }


def run_train_epoch(system: SimpleNamespace, train_loader, cfg):
    system.mol_encoder.train()
    system.sol_encoder.train()
    if getattr(system, "use_fusion_self_attention", False):
        system.fusion_encoder.train()
    system.mlp.train()

    full_train_eval_each_epoch = bool(getattr(cfg, "FULL_TRAIN_EVAL_EACH_EPOCH", False))

    pred_list: list[torch.Tensor] = []
    true_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor | None] = []

    for batch in train_loader:
        batch = move_batch_to_device(batch, system.device)

        x_vec = build_x_vec(batch, system)
        y_true = batch.y
        y_mask = getattr(batch, "y_mask", None)
        l2sp_loss = compute_l2sp_loss(system, cfg)

        out = train_step(
            model=system.mlp,
            x_vec=x_vec,
            y_true=y_true,
            optimizer=system.optimizer,
            y_mask=y_mask,
            extra_loss=l2sp_loss,
            clip_grad_norm=5.0,
            clip_grad_params=system.all_trainable_params,
            denom_eps=cfg.DENOM_EPS,
            require_nonzero_denom_for_loss=True,
            r2_require_nonzero_denom=False,
            auto_move_to_model_device=True,
        )

        pred_list.append(_to_2d_cpu_tensor(out["y_pred"]))
        true_list.append(_to_2d_cpu_tensor(out["y_true"]))
        mask_list.append(_to_2d_cpu_tensor(out["y_mask"]) if out["y_mask"] is not None else None)

    target_std, target_mean = _get_target_scale_and_bias(train_loader)

    if full_train_eval_each_epoch:
        return run_eval_epoch(system, train_loader, cfg)
    return _compute_full_loader_metrics(
        pred_list,
        true_list,
        mask_list,
        target_std=target_std,
        target_mean=target_mean,
    )


@torch.no_grad()
def run_eval_epoch(system: SimpleNamespace, loader, cfg):
    system.mol_encoder.eval()
    system.sol_encoder.eval()
    if getattr(system, "use_fusion_self_attention", False):
        system.fusion_encoder.eval()
    system.mlp.eval()

    pred_list: list[torch.Tensor] = []
    true_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor | None] = []

    for batch in loader:
        batch = move_batch_to_device(batch, system.device)

        x_vec = build_x_vec(batch, system)
        y_true = batch.y
        y_mask = getattr(batch, "y_mask", None)

        out = eval_step(
            model=system.mlp,
            x_vec=x_vec,
            y_true=y_true,
            y_mask=y_mask,
            denom_eps=cfg.DENOM_EPS,
            require_nonzero_denom_for_loss=True,
            r2_require_nonzero_denom=False,
            auto_move_to_model_device=True,
        )

        pred_list.append(_to_2d_cpu_tensor(out["y_pred"]))
        true_list.append(_to_2d_cpu_tensor(y_true))
        mask_list.append(_to_2d_cpu_tensor(y_mask) if y_mask is not None else None)

    target_std, target_mean = _get_target_scale_and_bias(loader)
    return _compute_full_loader_metrics(
        pred_list,
        true_list,
        mask_list,
        target_std=target_std,
        target_mean=target_mean,
    )


def _get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    if len(optimizer.param_groups) == 0:
        return float("nan")
    return float(optimizer.param_groups[0].get("lr", float("nan")))


def _step_scheduler(system: SimpleNamespace, *, val_loss: float, val_r2: float) -> None:
    scheduler = getattr(system, "scheduler", None)
    if scheduler is None:
        return

    monitor = str(getattr(system, "scheduler_monitor", "val_loss")).strip().lower()
    metric = val_r2 if monitor == "val_r2" else val_loss

    if not np.isfinite(metric):
        print(f"[LR Scheduler] skip step because monitored metric {monitor} is NaN/Inf.")
        return

    old_lr = _get_current_lr(system.optimizer)

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()

    new_lr = _get_current_lr(system.optimizer)
    if np.isfinite(old_lr) and np.isfinite(new_lr):
        if abs(new_lr - old_lr) > 1e-15:
            print(f"[LR Scheduler] {monitor}={metric:.6f} | lr: {old_lr:.8g} -> {new_lr:.8g}")
        else:
            print(f"[LR Scheduler] {monitor}={metric:.6f} | lr unchanged: {new_lr:.8g}")


def _build_checkpoint_cfg(system: SimpleNamespace, cfg) -> dict:
    keys = [
        "TRAIN_CSV",
        "VAL_CSV",
        "TEST_CSV",
        "CV_FOLD",
        "BATCH_SIZE",
        "EPOCHS",
        "MIN_EPOCHS",
        "LR",
        "WEIGHT_DECAY",
        "L2SP_ENABLED",
        "L2SP_LAMBDA",
        "L2SP_MODULES",
        "L2SP_EXCLUDE_BIAS_NORM",
        "DENOM_EPS",
        "SEED",
        "NUM_WORKERS",
        "PIN_MEMORY",
        "PERSISTENT_WORKERS",
        "PREFETCH_FACTOR",
        "CACHE_GRAPHS",
        "SOLVENT_MODE",
        "MORGAN_RADIUS",
        "MORGAN_N_BITS",
        "MORGAN_USE_CHIRALITY",
        "RDKIT_DESCRIPTOR_NAMES",
        "RDKIT_NAN_VALUE",
        "RDKIT_INF_VALUE",
        "GRAPH_TRANSFORMER_HIDDEN",
        "GRAPH_TRANSFORMER_LAYERS",
        "GRAPH_TRANSFORMER_OUT",
        "GRAPH_TRANSFORMER_HEADS",
        "GRAPH_TRANSFORMER_EDGE_DIM",
        "GRAPH_TRANSFORMER_DROPOUT",
        "GRAPH_TRANSFORMER_USE_EDGE_ATTR",
        "GRAPH_TRANSFORMER_BETA",
        "MLP_HIDDEN",
        "MLP_LAYERS",
        "MLP_DROPOUT",
        "EARLY_STOPPING_PATIENCE",
        "EARLY_STOPPING_MIN_DELTA",
        "EARLY_STOPPING_MODE",
        "EARLY_STOPPING_VERBOSE",
        "TARGET_NAMES",
        "FULL_TRAIN_EVAL_EACH_EPOCH",
        "LR_SCHEDULER_ENABLED",
        "LR_SCHEDULER_NAME",
        "LR_SCHEDULER_MONITOR",
        "LR_SCHEDULER_MODE",
        "LR_SCHEDULER_FACTOR",
        "LR_SCHEDULER_PATIENCE",
        "LR_SCHEDULER_THRESHOLD",
        "LR_SCHEDULER_THRESHOLD_MODE",
        "LR_SCHEDULER_COOLDOWN",
        "LR_SCHEDULER_MIN_LR",
        "LR_SCHEDULER_EPS",
        "SOLVENT_GRAPH_ENCODER_TYPE",
        "SOLVENT_GRAPH_TRANSFORMER_HIDDEN",
        "SOLVENT_GRAPH_TRANSFORMER_LAYERS",
        "SOLVENT_GRAPH_TRANSFORMER_OUT",
        "SOLVENT_GRAPH_TRANSFORMER_HEADS",
        "SOLVENT_GRAPH_TRANSFORMER_EDGE_DIM",
        "SOLVENT_GRAPH_TRANSFORMER_DROPOUT",
        "SOLVENT_GRAPH_TRANSFORMER_USE_EDGE_ATTR",
        "SOLVENT_GRAPH_TRANSFORMER_BETA",
        "SOLVENT_GAT_HIDDEN",
        "SOLVENT_GAT_LAYERS",
        "SOLVENT_GAT_OUT",
        "SOLVENT_GAT_HEADS",
        "SOLVENT_GAT_CONCAT",
        "SOLVENT_GAT_DROPOUT",
        "FUSION_TYPE",
        "FUSION_ATTENTION_DIM",
        "FUSION_ATTENTION_HEADS",
        "FUSION_ATTENTION_LAYERS",
        "FUSION_ATTENTION_FFN_DIM",
        "FUSION_ATTENTION_DROPOUT",
        "FUSION_OUTPUT_MODE",
    ]
    out = {k: getattr(cfg, k) for k in keys if hasattr(cfg, k)}
    out.update(
        {
            "MLP_INPUT_DIM": system.mlp_input_dim,
            "in_dim_mol": system.in_dim_mol,
            "in_dim_sol": getattr(system, "solvent_feat_dim", system.in_dim_sol),
            "use_solvent_graph_encoder": bool(getattr(system, "use_solvent_graph_encoder", False)),
            "use_solvent_edge_attr": bool(getattr(system, "use_solvent_edge_attr", False)),
            "use_fusion_self_attention": bool(getattr(system, "use_fusion_self_attention", False)),
            "fusion_output_dim": int(getattr(system, "fusion_output_dim", system.mlp_input_dim)),
        }
    )
    return out


def _save_training_log_compat(**kwargs) -> None:
    sig = inspect.signature(save_training_log)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    save_training_log(**filtered)


def _collect_invalid_smiles_stats(loader, prefix: str) -> dict:
    stats = getattr(getattr(loader, "dataset", None), "invalid_smiles_stats", None)
    if not isinstance(stats, dict):
        stats = {}
    return {
        f"{prefix}_total_rows_before": int(stats.get("total_rows_before", 0)),
        f"{prefix}_kept_rows": int(stats.get("kept_rows", 0)),
        f"{prefix}_dropped_total_rows": int(stats.get("dropped_total_rows", 0)),
        f"{prefix}_solute_skipped_rows": int(stats.get("solute_skipped_rows", 0)),
        f"{prefix}_solute_invalid_count": int(stats.get("solute_invalid_count", 0)),
        f"{prefix}_solute_missing_count": int(stats.get("solute_missing_count", 0)),
        f"{prefix}_solvent_smiles_required": int(stats.get("solvent_smiles_required", 0)),
        f"{prefix}_solvent_skipped_rows": int(stats.get("solvent_skipped_rows", 0)),
        f"{prefix}_solvent_invalid_count": int(stats.get("solvent_invalid_count", 0)),
        f"{prefix}_solvent_missing_count": int(stats.get("solvent_missing_count", 0)),
    }


def train_experiment(system: SimpleNamespace, train_loader, val_loader, test_loader, cfg) -> None:
    initialize_l2sp_regularizer(system, cfg)

    best_val_r2 = float("nan")
    best_epoch = 0

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    train_mae_history: list[float] = []
    val_mae_history: list[float] = []
    train_rmse_history: list[float] = []
    val_rmse_history: list[float] = []
    train_r2_history: list[float] = []
    val_r2_history: list[float] = []

    train_task_loss_history: list[np.ndarray] = []
    val_task_loss_history: list[np.ndarray] = []
    train_task_mae_history: list[np.ndarray] = []
    val_task_mae_history: list[np.ndarray] = []
    train_task_rmse_history: list[np.ndarray] = []
    val_task_rmse_history: list[np.ndarray] = []
    train_task_r2_history: list[np.ndarray] = []
    val_task_r2_history: list[np.ndarray] = []

    min_epochs = int(getattr(cfg, "MIN_EPOCHS", 200))
    if int(cfg.EPOCHS) < min_epochs:
        raise ValueError(
            f"cfg.EPOCHS={cfg.EPOCHS}  MIN_EPOCHS={min_epochs}，"
            f" {min_epochs}  epoch。"
        )

    for epoch in range(1, cfg.EPOCHS + 1):
        train_metrics = run_train_epoch(system, train_loader, cfg)
        val_metrics = run_eval_epoch(system, val_loader, cfg)

        train_loss = train_metrics["loss"]
        train_mae = train_metrics["mae"]
        train_rmse = train_metrics["rmse"]
        train_r2 = train_metrics["r2"]
        train_task_loss = train_metrics["loss_task"]
        train_task_mae = train_metrics["mae_task"]
        train_task_rmse = train_metrics["rmse_task"]
        train_task_r2 = train_metrics["r2_task"]

        val_loss = val_metrics["loss"]
        val_mae = val_metrics["mae"]
        val_rmse = val_metrics["rmse"]
        val_r2 = val_metrics["r2"]
        val_task_loss = val_metrics["loss_task"]
        val_task_mae = val_metrics["mae_task"]
        val_task_rmse = val_metrics["rmse_task"]
        val_task_r2 = val_metrics["r2_task"]

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_mae_history.append(train_mae)
        val_mae_history.append(val_mae)
        train_rmse_history.append(train_rmse)
        val_rmse_history.append(val_rmse)
        train_r2_history.append(train_r2)
        val_r2_history.append(val_r2)

        train_task_loss_history.append(train_task_loss.copy())
        val_task_loss_history.append(val_task_loss.copy())
        train_task_mae_history.append(train_task_mae.copy())
        val_task_mae_history.append(val_task_mae.copy())
        train_task_rmse_history.append(train_task_rmse.copy())
        val_task_rmse_history.append(val_task_rmse.copy())
        train_task_r2_history.append(train_task_r2.copy())
        val_task_r2_history.append(val_task_r2.copy())

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} train_mae={train_mae:.6f} train_rmse={train_rmse:.6f} train_r2={train_r2:.4f} | "
            f"val_loss={val_loss:.6f} val_mae={val_mae:.6f} val_rmse={val_rmse:.6f} val_r2={val_r2:.4f}"
        )

        print(
            "  Train task MAE:",
            ", ".join([f"{name}={train_task_mae[i]:.6f}" for i, name in enumerate(cfg.TARGET_NAMES)])
        )
        print(
            "  Train task RMSE:",
            ", ".join([f"{name}={train_task_rmse[i]:.6f}" for i, name in enumerate(cfg.TARGET_NAMES)])
        )
        print(
            "  Train task R2:",
            ", ".join([f"{name}={train_task_r2[i]:.4f}" for i, name in enumerate(cfg.TARGET_NAMES)])
        )
        print(
            "  Val   task MAE:",
            ", ".join([f"{name}={val_task_mae[i]:.6f}" for i, name in enumerate(cfg.TARGET_NAMES)])
        )
        print(
            "  Val   task RMSE:",
            ", ".join([f"{name}={val_task_rmse[i]:.6f}" for i, name in enumerate(cfg.TARGET_NAMES)])
        )
        print(
            "  Val   task R2:",
            ", ".join([f"{name}={val_task_r2[i]:.4f}" for i, name in enumerate(cfg.TARGET_NAMES)])
        )
        print(
            "  Train task Loss:",
            ", ".join([f"{name}={train_task_loss[i]:.6f}" for i, name in enumerate(cfg.TARGET_NAMES)])
        )
        print(
            "  Val   task Loss:",
            ", ".join([f"{name}={val_task_loss[i]:.6f}" for i, name in enumerate(cfg.TARGET_NAMES)])
        )

        _step_scheduler(system, val_loss=val_loss, val_r2=val_r2)
        print(f"  Current LR: {_get_current_lr(system.optimizer):.8g}")

        if np.isfinite(val_r2):
            should_stop = system.early_stopper.step(score=val_r2, epoch=epoch)

            if system.early_stopper.last_improved:
                best_val_r2 = float(system.early_stopper.best_score)
                best_epoch = int(system.early_stopper.best_epoch)

                torch.save(
                    {
                        "epoch": epoch,
                        "mol_encoder": system.mol_encoder.state_dict(),
                        "sol_encoder": system.sol_encoder.state_dict(),
                        "fusion_encoder": system.fusion_encoder.state_dict() if getattr(system, "use_fusion_self_attention", False) else None,
                        "mlp": system.mlp.state_dict(),
                        "optimizer": system.optimizer.state_dict(),
                        "scheduler": system.scheduler.state_dict() if getattr(system, "scheduler", None) is not None else None,
                        "val_r2": best_val_r2,
                        "device": str(system.device),
                        "cfg": _build_checkpoint_cfg(system, cfg),
                    },
                    cfg.BEST_PATH,
                )
                print(f"  [Best] saved -> {cfg.BEST_PATH} (val_r2={best_val_r2:.4f})")

            if should_stop and epoch >= min_epochs:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break
            elif should_stop and epoch < min_epochs:
                print(
                    f"[EarlyStopping] stop condition met at epoch {epoch}, "
                    f"but MIN_EPOCHS={min_epochs}, so continue training."
                )
        else:
            print(f"[EarlyStopping] skip epoch {epoch} because val_r2 is NaN or Inf.")

    print(f"\nTraining done. Best val_r2 = {best_val_r2:.4f}")
    print(f"Best checkpoint: {cfg.BEST_PATH}")

    test_loss = float("nan")
    test_mae = float("nan")
    test_rmse = float("nan")
    test_r2 = float("nan")
    test_task_loss = np.full((TARGET_DIM,), np.nan, dtype=float)
    test_task_mae = np.full((TARGET_DIM,), np.nan, dtype=float)
    test_task_rmse = np.full((TARGET_DIM,), np.nan, dtype=float)
    test_task_r2 = np.full((TARGET_DIM,), np.nan, dtype=float)

    if Path(cfg.BEST_PATH).exists():
        best_ckpt = torch.load(cfg.BEST_PATH, map_location=system.device)
        system.mol_encoder.load_state_dict(best_ckpt["mol_encoder"])
        if "sol_encoder" in best_ckpt:
            system.sol_encoder.load_state_dict(best_ckpt["sol_encoder"], strict=False)
        if getattr(system, "use_fusion_self_attention", False) and best_ckpt.get("fusion_encoder") is not None:
            system.fusion_encoder.load_state_dict(best_ckpt["fusion_encoder"], strict=False)
        system.mlp.load_state_dict(best_ckpt["mlp"])

        test_metrics = run_eval_epoch(system, test_loader, cfg)
        test_loss = test_metrics["loss"]
        test_mae = test_metrics["mae"]
        test_rmse = test_metrics["rmse"]
        test_r2 = test_metrics["r2"]
        test_task_loss = test_metrics["loss_task"]
        test_task_mae = test_metrics["mae_task"]
        test_task_rmse = test_metrics["rmse_task"]
        test_task_r2 = test_metrics["r2_task"]
        print(f"\n[Test] loss={test_loss:.6f} test_mae={test_mae:.6f} test_rmse={test_rmse:.6f} test_r2={test_r2:.4f}")
        print(
            "  Test  task MAE:",
            ", ".join([f"{name}={test_task_mae[i]:.6f}" for i, name in enumerate(cfg.TARGET_NAMES)])
        )
        print(
            "  Test  task RMSE:",
            ", ".join([f"{name}={test_task_rmse[i]:.6f}" for i, name in enumerate(cfg.TARGET_NAMES)])
        )
        print(
            "  Test  task R2:",
            ", ".join([f"{name}={test_task_r2[i]:.4f}" for i, name in enumerate(cfg.TARGET_NAMES)])
        )
        print(
            "  Test  task Loss:",
            ", ".join([f"{name}={test_task_loss[i]:.6f}" for i, name in enumerate(cfg.TARGET_NAMES)])
        )

    invalid_stats_kwargs = {}
    invalid_stats_kwargs.update(_collect_invalid_smiles_stats(train_loader, "train"))
    invalid_stats_kwargs.update(_collect_invalid_smiles_stats(val_loader, "val"))
    invalid_stats_kwargs.update(_collect_invalid_smiles_stats(test_loader, "test"))

    _save_training_log_compat(
        log_path=cfg.LOG_PATH,
        log_csv_path=cfg.LOG_CSV_PATH,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        train_mae_history=train_mae_history,
        val_mae_history=val_mae_history,
        train_rmse_history=train_rmse_history,
        val_rmse_history=val_rmse_history,
        train_r2_history=train_r2_history,
        val_r2_history=val_r2_history,
        train_task_loss_history=train_task_loss_history,
        val_task_loss_history=val_task_loss_history,
        train_task_mae_history=train_task_mae_history,
        val_task_mae_history=val_task_mae_history,
        train_task_rmse_history=train_task_rmse_history,
        val_task_rmse_history=val_task_rmse_history,
        train_task_r2_history=train_task_r2_history,
        val_task_r2_history=val_task_r2_history,
        best_epoch=best_epoch,
        **invalid_stats_kwargs,
    )

    return {
        "best_epoch": int(best_epoch),
        "best_val_r2": float(best_val_r2),
        "final_test_loss": float(test_loss),
        "final_test_mae": float(test_mae),
        "final_test_rmse": float(test_rmse),
        "final_test_r2": float(test_r2),
        "final_test_abs_loss": float(test_task_loss[0]),
        "final_test_emi_loss": float(test_task_loss[1]),
        "final_test_plqy_loss": float(test_task_loss[2]),
        "final_test_em_loss": float(test_task_loss[3]),
        "final_test_abs_mae": float(test_task_mae[0]),
        "final_test_emi_mae": float(test_task_mae[1]),
        "final_test_plqy_mae": float(test_task_mae[2]),
        "final_test_em_mae": float(test_task_mae[3]),
        "final_test_abs_rmse": float(test_task_rmse[0]),
        "final_test_emi_rmse": float(test_task_rmse[1]),
        "final_test_plqy_rmse": float(test_task_rmse[2]),
        "final_test_em_rmse": float(test_task_rmse[3]),
        "final_test_abs_r2": float(test_task_r2[0]),
        "final_test_emi_r2": float(test_task_r2[1]),
        "final_test_plqy_r2": float(test_task_r2[2]),
        "final_test_em_r2": float(test_task_r2[3]),
    }

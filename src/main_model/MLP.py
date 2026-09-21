from __future__ import annotations

import copy
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn


TARGET_NAMES = ["abs", "emi", "plqy", "em"]





def get_preferred_device(explicit_device: Optional[str | torch.device] = None) -> torch.device:
    """
     CUDA；，。
    """
    if explicit_device is not None:
        return torch.device(explicit_device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def get_model_device(model: nn.Module) -> torch.device:
    """
     device。
    ，。
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return get_preferred_device()



def get_model_dtype(model: nn.Module) -> torch.dtype:
    """
     dtype。
    ， float32。
    """
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32



def move_model_to_preferred_device(
    model: nn.Module,
    device: Optional[str | torch.device] = None,
) -> Tuple[nn.Module, torch.device]:
    """
    （ CUDA ）。
    ：model, device
    """
    target_device = get_preferred_device(device)
    model = model.to(target_device)
    return model, target_device



def to_device_tensor(
    x,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = True,
) -> Optional[torch.Tensor]:
    """
     tensor / numpy / list / scalar  device。
    x  None  None。
    """
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        if dtype is None:
            return x.to(device=device, non_blocking=non_blocking)
        return x.to(device=device, dtype=dtype, non_blocking=non_blocking)

    return torch.as_tensor(x, device=device, dtype=dtype)



def to_model_device(
    model: nn.Module,
    x,
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = True,
) -> Optional[torch.Tensor]:
    """
     model  device。
    """
    device = get_model_device(model)
    return to_device_tensor(x, device=device, dtype=dtype, non_blocking=non_blocking)



def new_tensor_like(
    ref_tensor: torch.Tensor,
    data,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
     ref_tensor  device（ dtype） tensor。
    """
    if dtype is None:
        dtype = ref_tensor.dtype
    return torch.as_tensor(data, device=ref_tensor.device, dtype=dtype)





def init_mlp_weights(m: nn.Module) -> None:
    """
    Linear  Xavier ；bias=0。
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLP(nn.Module):
    """
    :
      x: [B, input_dim]
    :
      y_pred: [B, 4] -> [abs, emi, plqy, em]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        layers: int = 3,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        activation: Optional[nn.Module] = None,
        out_dim: int = 4,
        auto_to_device: bool = True,
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()

        if layers < 1:
            raise ValueError("layers  >= 1")

        if activation is None:
            activation = nn.ReLU()

        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        self.auto_to_device = bool(auto_to_device)

        mods: List[nn.Module] = []

        mods.append(nn.Linear(self.input_dim, hidden_dim))
        mods.append(copy.deepcopy(activation))
        if use_layernorm:
            mods.append(nn.LayerNorm(hidden_dim))
        mods.append(nn.Dropout(dropout))

        for _ in range(layers - 1):
            mods.append(nn.Linear(hidden_dim, hidden_dim))
            mods.append(copy.deepcopy(activation))
            if use_layernorm:
                mods.append(nn.LayerNorm(hidden_dim))
            mods.append(nn.Dropout(dropout))

        mods.append(nn.Linear(hidden_dim, self.out_dim))

        self.net = nn.Sequential(*mods)
        self.net.apply(init_mlp_weights)

        if self.auto_to_device:
            target_device = get_preferred_device(device)
            self.to(target_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"MLP.forward  torch.Tensor， {type(x)}")

        if x.ndim != 2:
            raise ValueError(f"MLP  [B,D]， {tuple(x.shape)}")

        if x.size(1) != self.input_dim:
            raise ValueError(f"MLP ： D={self.input_dim}， D={x.size(1)}")

        model_device = get_model_device(self)
        model_dtype = get_model_dtype(self)

        if x.device != model_device or x.dtype != model_dtype:
            x = x.to(device=model_device, dtype=model_dtype, non_blocking=True)

        return self.net(x)


class MultiTaskMLP(nn.Module):
    """
     + 4  head。

    :
      x: [B, input_dim]
    :
      y_pred: [B, out_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        layers: int = 3,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        activation: Optional[nn.Module] = None,
        out_dim: int = 4,
        auto_to_device: bool = True,
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()

        if layers < 1:
            raise ValueError("layers  >= 1")

        if activation is None:
            activation = nn.ReLU()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.auto_to_device = bool(auto_to_device)

        trunk: List[nn.Module] = []
        trunk.append(nn.Linear(self.input_dim, self.hidden_dim))
        trunk.append(copy.deepcopy(activation))
        if use_layernorm:
            trunk.append(nn.LayerNorm(self.hidden_dim))
        trunk.append(nn.Dropout(dropout))

        for _ in range(layers - 1):
            trunk.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            trunk.append(copy.deepcopy(activation))
            if use_layernorm:
                trunk.append(nn.LayerNorm(self.hidden_dim))
            trunk.append(nn.Dropout(dropout))

        self.trunk = nn.Sequential(*trunk)
        self.task_heads = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in range(self.out_dim)])

        self.trunk.apply(init_mlp_weights)
        self.task_heads.apply(init_mlp_weights)

        if self.auto_to_device:
            target_device = get_preferred_device(device)
            self.to(target_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"MultiTaskMLP.forward  torch.Tensor， {type(x)}")

        if x.ndim != 2:
            raise ValueError(f"MultiTaskMLP  [B,D]， {tuple(x.shape)}")

        if x.size(1) != self.input_dim:
            raise ValueError(f"MultiTaskMLP ： D={self.input_dim}， D={x.size(1)}")

        model_device = get_model_device(self)
        model_dtype = get_model_dtype(self)

        if x.device != model_device or x.dtype != model_dtype:
            x = x.to(device=model_device, dtype=model_dtype, non_blocking=True)

        shared = self.trunk(x)
        outputs = [head(shared) for head in self.task_heads]
        return torch.cat(outputs, dim=1)





def _ensure_2d_y(y: torch.Tensor) -> torch.Tensor:
    """
     y  [B,4]  [B,1,4]， [B,4]
    """
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"y  torch.Tensor， {type(y)}")

    if y.ndim == 3 and y.size(1) == 1:
        y = y.view(y.size(0), y.size(-1))

    if y.ndim != 2 or y.size(1) != 4:
        raise ValueError(f" y  [B,4]， {tuple(y.shape)}")

    return y



def _prepare_xy_mask_for_model(
    model: nn.Module,
    x_vec: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
     x_vec / y_true / y_mask  device。
    """
    model_device = get_model_device(model)
    model_dtype = get_model_dtype(model)

    x_vec = to_device_tensor(x_vec, device=model_device, dtype=model_dtype)
    y_true = to_device_tensor(y_true, device=model_device, dtype=torch.float32)

    if y_mask is not None:
        y_mask = to_device_tensor(y_mask, device=model_device, dtype=torch.float32)

    y_true = _ensure_2d_y(y_true)
    if y_mask is not None:
        y_mask = _ensure_2d_y(y_mask)

    return x_vec, y_true, y_mask



def make_effective_mask(
    y_true: torch.Tensor,
    y_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
     mask（True=，False=）：
      -  y_mask ： y_mask==1
      -  y_true  (not NaN/Inf)
    """
    y_true = _ensure_2d_y(y_true).float()

    if y_mask is None:
        base = torch.ones_like(y_true, dtype=torch.bool, device=y_true.device)
    else:
        y_mask = _ensure_2d_y(y_mask).float()
        base = y_mask > 0.5

    finite = torch.isfinite(y_true)
    return base & finite





def masked_mse_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    y_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
     masked MSE：
      -  mask=1  y 
      -  NaN  0 ， eff_mask 
     loss。
    """
    pred = _ensure_2d_y(pred).float()
    y = _ensure_2d_y(y).float()

    if pred.device != y.device:
        raise RuntimeError(f"pred  y  device：{pred.device} vs {y.device}")

    if y_mask is None:
        y_mask = torch.ones_like(y, dtype=torch.float32, device=y.device)
    else:
        y_mask = _ensure_2d_y(y_mask).float()
        if y_mask.device != y.device:
            raise RuntimeError(f"y_mask  y  device：{y_mask.device} vs {y.device}")

    finite = torch.isfinite(y).float()
    eff_mask = y_mask * finite
    y_safe = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    loss_mat = (pred - y_safe).pow(2)
    denom = eff_mask.sum().clamp(min=1.0)
    return (loss_mat * eff_mask).sum() / denom



def masked_mse_loss_with_stats(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: Optional[torch.Tensor] = None,
    per_task_average: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ：
      loss:  tensor
      loss_task: [4]  loss（ nan）
      sq_err: [B,4] （）
      eff_mask: [B,4]  mask（float）
    """
    y_pred = _ensure_2d_y(y_pred).float()
    y_true = _ensure_2d_y(y_true).float()

    if y_pred.device != y_true.device:
        raise RuntimeError(f"y_pred  y_true  device：{y_pred.device} vs {y_true.device}")

    if y_mask is None:
        y_mask = torch.ones_like(y_true, dtype=torch.float32, device=y_true.device)
    else:
        y_mask = _ensure_2d_y(y_mask).float()

    finite = torch.isfinite(y_true).float()
    eff_mask = y_mask * finite
    y_safe = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    sq_err = (y_pred - y_safe).pow(2)

    loss_task_list: List[torch.Tensor] = []
    for j in range(y_true.size(1)):
        m = eff_mask[:, j] > 0.5
        if m.any():
            loss_task_list.append(sq_err[m, j].mean())
        else:
            loss_task_list.append(torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype))

    loss_task = torch.stack(loss_task_list)

    if per_task_average:
        valid = ~torch.isnan(loss_task)
        if valid.any():
            loss = loss_task[valid].mean()
        else:
            loss = torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype)
    else:
        denom = eff_mask.sum().clamp(min=1.0)
        if denom.item() > 0:
            loss = (sq_err * eff_mask).sum() / denom
        else:
            loss = torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype)

    return loss, loss_task, sq_err, eff_mask



def mse_loss_ignore_masked(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: Optional[torch.Tensor] = None,
    per_task_average: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    。
    """
    return masked_mse_loss_with_stats(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        per_task_average=per_task_average,
    )



def masked_mse_loss_ignore_masked(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: Optional[torch.Tensor] = None,
    denom_eps: float = 1e-8,
    require_nonzero_denom: bool = True,
    per_task_average: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
     masked MSE 。
     masked MSE：
      -  mask=1  y_true 
      - denom_eps / require_nonzero_denom ， masked MSE 
    。
    """
    del denom_eps, require_nonzero_denom
    return masked_mse_loss_with_stats(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        per_task_average=per_task_average,
    )


def relative_mse_loss_ignore_masked(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: Optional[torch.Tensor] = None,
    denom_eps: float = 1e-8,
    require_nonzero_denom: bool = True,
    per_task_average: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
     masked relative MSE：
      relative_sq_err = (y_pred - y_true)^2 / max(y_true^2, denom_eps)

    ：
      - y_mask == 1
      - y_true 
      -  require_nonzero_denom=True ， |y_true| > denom_eps

     masked_mse_loss_with_stats ：
      loss, loss_task, relative_sq_err, eff_mask
    """
    y_pred = _ensure_2d_y(y_pred).float()
    y_true = _ensure_2d_y(y_true).float()

    if y_pred.device != y_true.device:
        raise RuntimeError(f"y_pred  y_true  device：{y_pred.device} vs {y_true.device}")

    if y_mask is None:
        y_mask = torch.ones_like(y_true, dtype=torch.float32, device=y_true.device)
    else:
        y_mask = _ensure_2d_y(y_mask).float()
        if y_mask.device != y_true.device:
            raise RuntimeError(f"y_mask  y_true  device：{y_mask.device} vs {y_true.device}")

    finite = torch.isfinite(y_true).float()
    eff_mask = y_mask * finite

    abs_y = torch.abs(y_true)
    if require_nonzero_denom:
        eff_mask = eff_mask * (abs_y > float(denom_eps)).float()

    y_safe = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    denom = torch.clamp(y_safe.pow(2), min=float(denom_eps))
    relative_sq_err = (y_pred - y_safe).pow(2) / denom

    loss_task_list: List[torch.Tensor] = []
    for j in range(y_true.size(1)):
        m = eff_mask[:, j] > 0.5
        if m.any():
            loss_task_list.append(relative_sq_err[m, j].mean())
        else:
            loss_task_list.append(torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype))

    loss_task = torch.stack(loss_task_list)

    if per_task_average:
        valid = ~torch.isnan(loss_task)
        if valid.any():
            loss = loss_task[valid].mean()
        else:
            loss = torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype)
    else:
        denom_all = eff_mask.sum().clamp(min=1.0)
        if denom_all.item() > 0:
            loss = (relative_sq_err * eff_mask).sum() / denom_all
        else:
            loss = torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype)

    return loss, loss_task, relative_sq_err, eff_mask





def r2_ignore_masked(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    require_nonzero_denom: bool = False,
    denom_eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
     R²: 1 - SS_res / SS_tot
    （ masked； NaN/Inf； |y|<=denom_eps）
    """
    y_pred = _ensure_2d_y(y_pred).float()
    y_true = _ensure_2d_y(y_true).float()

    if y_pred.device != y_true.device:
        raise RuntimeError(f"y_pred  y_true  device：{y_pred.device} vs {y_true.device}")

    eff_mask = make_effective_mask(y_true=y_true, y_mask=y_mask)
    if require_nonzero_denom:
        eff_mask = eff_mask & (torch.abs(y_true) > float(denom_eps))

    r2_list: List[torch.Tensor] = []
    for j in range(y_true.size(1)):
        m = eff_mask[:, j]
        if int(m.sum().item()) < 2:
            r2_list.append(torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype))
            continue

        yt = y_true[m, j]
        yp = y_pred[m, j]

        ss_res = (yp - yt).pow(2).sum()
        ss_tot = (yt - yt.mean()).pow(2).sum()

        if ss_tot < float(eps):
            r2_list.append(torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype))
        else:
            r2_list.append(1.0 - ss_res / (ss_tot + float(eps)))

    r2_task = torch.stack(r2_list)
    valid = ~torch.isnan(r2_task)

    if valid.any():
        r2_mean = r2_task[valid].mean()
    else:
        r2_mean = torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype)

    return r2_mean, r2_task, eff_mask.float()





@torch.no_grad()
def _detach_to_cpu(x: torch.Tensor) -> Any:
    return x.detach().cpu()





def train_step(
    model: nn.Module,
    x_vec: torch.Tensor,
    y_true: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    y_mask: Optional[torch.Tensor] = None,
    extra_loss: Optional[torch.Tensor] = None,
    clip_grad_norm: Optional[float] = 5.0,
    clip_grad_params = None,
    denom_eps: float = 1e-8,
    require_nonzero_denom_for_loss: bool = True,
    r2_require_nonzero_denom: bool = False,
    auto_move_to_model_device: bool = True,
) -> Dict[str, Any]:
    """
     batch ：
      -  x_vec / y_true / y_mask  model  device
      -  masked MSE Loss（ masked ）
      - R²（ masked）

    ：
      - denom_eps / require_nonzero_denom_for_loss ，
         masked MSE  loss 。
    """
    del denom_eps, require_nonzero_denom_for_loss
    model.train()

    if auto_move_to_model_device:
        x_vec, y_true, y_mask = _prepare_xy_mask_for_model(model, x_vec, y_true, y_mask)
    else:
        y_true = _ensure_2d_y(y_true).float()
        if y_mask is not None:
            y_mask = _ensure_2d_y(y_mask).float()

    y_pred = model(x_vec)

    loss, loss_task, _, eff_mask_loss = masked_mse_loss_ignore_masked(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        per_task_average=False,
    )

    if (eff_mask_loss.sum() <= 0) or torch.isnan(loss):
        with torch.no_grad():
            r2_mean, r2_task, eff_mask_r2 = r2_ignore_masked(
                y_pred=y_pred,
                y_true=y_true,
                y_mask=y_mask,
                require_nonzero_denom=r2_require_nonzero_denom,
            )

        return {
            "skipped": True,
            "device": str(get_model_device(model)),
            "loss": float("nan"),
            "loss_task": _detach_to_cpu(loss_task).numpy(),
            "r2_mean": float(r2_mean.item()) if not torch.isnan(r2_mean) else float("nan"),
            "r2_task": _detach_to_cpu(r2_task).numpy(),
            "n_valid_loss": int(eff_mask_loss.sum().item()),
            "n_valid_r2": int(eff_mask_r2.sum().item()),
            "y_pred": _detach_to_cpu(y_pred),
            "y_true": _detach_to_cpu(y_true),
            "y_mask": _detach_to_cpu(y_mask) if y_mask is not None else None,
        }

    if extra_loss is not None:
        extra_loss = extra_loss.to(device=loss.device, dtype=loss.dtype)
        total_loss = loss + extra_loss
    else:
        total_loss = loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    if clip_grad_norm is not None:
        params_to_clip = clip_grad_params if clip_grad_params is not None else model.parameters()
        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=float(clip_grad_norm))

    optimizer.step()

    with torch.no_grad():
        r2_mean, r2_task, eff_mask_r2 = r2_ignore_masked(
            y_pred=y_pred,
            y_true=y_true,
            y_mask=y_mask,
            require_nonzero_denom=r2_require_nonzero_denom,
        )

    return {
        "skipped": False,
        "device": str(get_model_device(model)),
        "loss": float(loss.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
        "extra_loss": float(extra_loss.detach().cpu().item()) if extra_loss is not None else 0.0,
        "loss_task": _detach_to_cpu(loss_task).numpy(),
        "r2_mean": float(r2_mean.detach().cpu().item()) if not torch.isnan(r2_mean) else float("nan"),
        "r2_task": _detach_to_cpu(r2_task).numpy(),
        "n_valid_loss": int(eff_mask_loss.sum().item()),
        "n_valid_r2": int(eff_mask_r2.sum().item()),
        "y_pred": _detach_to_cpu(y_pred),
        "y_true": _detach_to_cpu(y_true),
        "y_mask": _detach_to_cpu(y_mask) if y_mask is not None else None,
    }





@torch.no_grad()
def eval_step(
    model: nn.Module,
    x_vec: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: Optional[torch.Tensor] = None,
    denom_eps: float = 1e-8,
    require_nonzero_denom_for_loss: bool = True,
    r2_require_nonzero_denom: bool = False,
    auto_move_to_model_device: bool = True,
) -> Dict[str, Any]:
    """
    /：
      -  x_vec / y_true / y_mask  model  device
      -  masked MSE  R²， backward
    """
    del denom_eps, require_nonzero_denom_for_loss
    model.eval()

    if auto_move_to_model_device:
        x_vec, y_true, y_mask = _prepare_xy_mask_for_model(model, x_vec, y_true, y_mask)
    else:
        y_true = _ensure_2d_y(y_true).float()
        if y_mask is not None:
            y_mask = _ensure_2d_y(y_mask).float()

    y_pred = model(x_vec)

    loss, loss_task, _, eff_mask_loss = masked_mse_loss_ignore_masked(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        per_task_average=False,
    )

    r2_mean, r2_task, eff_mask_r2 = r2_ignore_masked(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        require_nonzero_denom=r2_require_nonzero_denom,
    )

    return {
        "device": str(get_model_device(model)),
        "loss": float(loss.detach().cpu().item()) if not torch.isnan(loss) else float("nan"),
        "loss_task": _detach_to_cpu(loss_task).numpy(),
        "r2_mean": float(r2_mean.detach().cpu().item()) if not torch.isnan(r2_mean) else float("nan"),
        "r2_task": _detach_to_cpu(r2_task).numpy(),
        "n_valid_loss": int(eff_mask_loss.sum().item()),
        "n_valid_r2": int(eff_mask_r2.sum().item()),
        "y_pred": _detach_to_cpu(y_pred),
        "y_true": _detach_to_cpu(y_true),
        "y_mask": _detach_to_cpu(y_mask) if y_mask is not None else None,
    }





def build_mlp(
    input_dim: int,
    hidden_dim: int = 512,
    layers: int = 3,
    dropout: float = 0.1,
    use_layernorm: bool = True,
    activation: Optional[nn.Module] = None,
    out_dim: int = 4,
    device: Optional[str | torch.device] = None,
) -> Tuple[MLP, torch.device]:
    """
    ：
      model, device = build_mlp(...)
    """
    target_device = get_preferred_device(device)
    model = MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        layers=layers,
        dropout=dropout,
        use_layernorm=use_layernorm,
        activation=activation,
        out_dim=out_dim,
        auto_to_device=False,
    )
    model = model.to(target_device)
    return model, target_device

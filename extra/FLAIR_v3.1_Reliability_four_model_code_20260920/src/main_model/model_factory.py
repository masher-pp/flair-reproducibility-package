from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from MLP import MultiTaskMLP
from define import FlexibleTransformerEncoder, FlexibleGATEncoder
from early_stopping import EarlyStopping





def get_preferred_device(explicit_device: str | torch.device | None = None) -> torch.device:
    if explicit_device is not None:
        return torch.device(explicit_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_module_device(module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return get_preferred_device()


def ensure_same_device(*modules) -> torch.device:
    devices = [get_module_device(m) for m in modules]
    first = devices[0]
    for i, d in enumerate(devices[1:], start=1):
        if d != first:
            raise RuntimeError(f" device ： 0  {first}， {i}  {d}")
    return first





class SoluteSolventSelfAttentionFusion(nn.Module):
    """
     tokens  tokens  token-level Self-Attention。

    ：
    1) ； return_node_embeddings=True  tokens。
    2)  CLS token。
    3) Self-Attention  CLS token  interaction_vec。
    4)  MLP  trainer.py ：
       concat(solute_vec, solvent_vec, interaction_vec)。
    """

    def __init__(
        self,
        solute_token_dim: int,
        solvent_token_dim: int,
        attention_dim: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 1,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        output_mode: str = "cls",
    ):
        super().__init__()

        self.solute_token_dim = int(solute_token_dim)
        self.solvent_token_dim = int(solvent_token_dim)
        self.attention_dim = int(attention_dim)
        self.attention_heads = int(attention_heads)
        self.attention_layers = int(attention_layers)
        self.dropout = float(dropout)
        self.output_mode = str(output_mode).strip().lower()

        if self.attention_dim <= 0:
            raise ValueError("fusion attention_dim  > 0")
        if self.attention_heads <= 0:
            raise ValueError("fusion attention_heads  > 0")
        if self.attention_layers <= 0:
            raise ValueError("fusion attention_layers  > 0")
        if self.attention_dim % self.attention_heads != 0:
            raise ValueError(
                f"fusion attention_dim={self.attention_dim}  "
                f"attention_heads={self.attention_heads} 。"
            )

        
        
        if self.output_mode in {"flatten", "flat"}:
            self.output_mode = "cls"
        if self.output_mode not in {"cls", "mean", "masked_mean"}:
            raise ValueError("token-level fusion_output_mode  'cls' / 'mean' / 'masked_mean'")

        if ffn_dim is None:
            ffn_dim = self.attention_dim * 4
        self.ffn_dim = int(ffn_dim)

        self.solute_proj = nn.Sequential(
            nn.Linear(self.solute_token_dim, self.attention_dim),
            nn.LayerNorm(self.attention_dim),
            nn.ReLU(),
        )
        self.solvent_proj = nn.Sequential(
            nn.Linear(self.solvent_token_dim, self.attention_dim),
            nn.LayerNorm(self.attention_dim),
            nn.ReLU(),
        )

        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.attention_dim))
        self.solute_type_embedding = nn.Parameter(torch.zeros(1, 1, self.attention_dim))
        self.solvent_type_embedding = nn.Parameter(torch.zeros(1, 1, self.attention_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.solute_type_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.solvent_type_embedding, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.attention_dim,
            nhead=self.attention_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.attention_layers)

        self.out_dim = self.attention_dim
        self.out_norm = nn.LayerNorm(self.out_dim)

    @staticmethod
    def _as_1d_batch(batch: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(batch, torch.Tensor):
            batch = torch.as_tensor(batch, dtype=torch.long)
        batch = batch.long().view(-1)
        if batch.ndim != 1:
            raise ValueError(f"{name}  batch ")
        return batch

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        
        mask_f = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return (tokens * mask_f).sum(dim=1) / denom

    def forward(
        self,
        solute_tokens: torch.Tensor,
        solute_batch: torch.Tensor,
        solvent_tokens: torch.Tensor,
        solvent_batch: torch.Tensor,
    ) -> torch.Tensor:
        if solute_tokens.ndim != 2:
            raise ValueError(f"solute_tokens  [N_solute,D]， {tuple(solute_tokens.shape)}")
        if solvent_tokens.ndim != 2:
            raise ValueError(f"solvent_tokens  [N_solvent,D]， {tuple(solvent_tokens.shape)}")

        solute_batch = self._as_1d_batch(solute_batch, "solute_batch").to(solute_tokens.device)
        solvent_batch = self._as_1d_batch(solvent_batch, "solvent_batch").to(solvent_tokens.device)

        if solute_tokens.size(0) != solute_batch.numel():
            raise ValueError(
                f"solute_tokens  solute_batch ："
                f"{solute_tokens.size(0)} vs {solute_batch.numel()}"
            )
        if solvent_tokens.size(0) != solvent_batch.numel():
            raise ValueError(
                f"solvent_tokens  solvent_batch ："
                f"{solvent_tokens.size(0)} vs {solvent_batch.numel()}"
            )

        solute_tokens = self.solute_proj(solute_tokens)
        solvent_tokens = self.solvent_proj(solvent_tokens)

        solute_dense, solute_mask = to_dense_batch(solute_tokens, solute_batch)
        solvent_dense, solvent_mask = to_dense_batch(solvent_tokens, solvent_batch)

        solute_dense = solute_dense + self.solute_type_embedding.to(
            device=solute_dense.device,
            dtype=solute_dense.dtype,
        )
        solvent_dense = solvent_dense + self.solvent_type_embedding.to(
            device=solvent_dense.device,
            dtype=solvent_dense.dtype,
        )

        tokens = torch.cat([solute_dense, solvent_dense], dim=1)
        mask = torch.cat([solute_mask, solvent_mask], dim=1)

        if self.output_mode == "cls":
            cls = self.cls_token.to(device=tokens.device, dtype=tokens.dtype).expand(tokens.size(0), -1, -1)
            cls_mask = torch.ones((tokens.size(0), 1), dtype=torch.bool, device=mask.device)
            tokens = torch.cat([cls, tokens], dim=1)
            mask = torch.cat([cls_mask, mask], dim=1)

        
        tokens = self.encoder(tokens, src_key_padding_mask=~mask)

        if self.output_mode == "cls":
            interaction_vec = tokens[:, 0, :]
        else:
            interaction_vec = self._masked_mean(tokens, mask)

        return self.out_norm(interaction_vec)


class TokenSelfAttentionGraphReadoutEncoder(nn.Module):
    """
     Graph Transformer 。

     graph_encoder  node tokens；
     atom tokens  dense ， CLS token，
     TransformerEncoder， CLS  graph_emb。
    """

    def __init__(
        self,
        graph_encoder: FlexibleTransformerEncoder,
        attention_heads: int = 4,
        attention_layers: int = 1,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.hidden_dims = getattr(graph_encoder, "hidden_dims", None)
        if self.hidden_dims is None or len(self.hidden_dims) == 0:
            raise ValueError("graph_encoder  hidden_dims， token-level self-attention readout。")

        self.token_dim = int(self.hidden_dims[-1])
        self.graph_dim = int(getattr(graph_encoder, "graph_dim", self.token_dim))
        self.attention_heads = int(attention_heads)
        self.attention_layers = int(attention_layers)
        self.dropout = float(dropout)

        if self.attention_heads <= 0:
            raise ValueError("GRAPH_TRANSFORMER_READOUT_HEADS  > 0")
        if self.attention_layers <= 0:
            raise ValueError("GRAPH_TRANSFORMER_READOUT_LAYERS  > 0")
        if self.token_dim % self.attention_heads != 0:
            raise ValueError(
                f" token_dim={self.token_dim}  "
                f"GRAPH_TRANSFORMER_READOUT_HEADS={self.attention_heads} 。"
            )

        if ffn_dim is None:
            ffn_dim = self.token_dim * 4
        self.ffn_dim = int(ffn_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=self.attention_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.readout_encoder = nn.TransformerEncoder(layer, num_layers=self.attention_layers)
        self.out_norm = nn.LayerNorm(self.token_dim)
        self.out_proj = nn.Linear(self.token_dim, self.graph_dim) if self.graph_dim != self.token_dim else nn.Identity()

    def forward(
        self,
        data=None,
        x=None,
        edge_index=None,
        batch=None,
        edge_attr=None,
        return_node_embeddings: bool = False,
    ):
        
        
        _, node_emb = self.graph_encoder(
            data=data,
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
            return_node_embeddings=True,
        )

        if data is not None:
            batch = getattr(data, "batch", batch)
        if batch is None:
            batch = torch.zeros(node_emb.size(0), dtype=torch.long, device=node_emb.device)
        batch = batch.long().view(-1).to(node_emb.device)

        if node_emb.size(0) != batch.numel():
            raise ValueError(
                f"node_emb  batch ：{node_emb.size(0)} vs {batch.numel()}"
            )

        dense_tokens, mask = to_dense_batch(node_emb, batch)
        cls = self.cls_token.to(device=dense_tokens.device, dtype=dense_tokens.dtype).expand(
            dense_tokens.size(0), -1, -1
        )
        cls_mask = torch.ones((dense_tokens.size(0), 1), dtype=torch.bool, device=mask.device)

        tokens = torch.cat([cls, dense_tokens], dim=1)
        mask = torch.cat([cls_mask, mask], dim=1)

        tokens = self.readout_encoder(tokens, src_key_padding_mask=~mask)
        graph_emb = self.out_proj(self.out_norm(tokens[:, 0, :]))

        if return_node_embeddings:
            return graph_emb, node_emb
        return graph_emb





def _infer_solvent_feat_dim(sample_batch) -> int:
    if hasattr(sample_batch, "solvent_feat"):
        feat = sample_batch.solvent_feat
    elif hasattr(sample_batch, "solv_cond"):
        feat = sample_batch.solv_cond
    else:
        raise ValueError("sample_batch  solvent_feat / solv_cond，。")

    if feat.ndim == 1:
        return int(feat.numel())
    return int(feat.size(-1))


def _infer_solvent_graph_in_dim(sample_batch) -> int:
    if not hasattr(sample_batch, "solvent_x"):
        raise ValueError("sample_batch  solvent_x，。")
    return int(sample_batch.solvent_x.size(1))





def _safe_torch_load(path: str | Path, map_location: torch.device):
    """ PyTorch  .pth。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


def _extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    """ checkpoint  state_dict。"""
    if hasattr(obj, "state_dict"):
        return obj.state_dict()

    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "gnn", "encoder"]:
            if key in obj:
                value = obj[key]
                if hasattr(value, "state_dict"):
                    return value.state_dict()
                if isinstance(value, dict):
                    return value

        tensor_like = [torch.is_tensor(v) for v in obj.values()]
        if len(tensor_like) > 0 and sum(tensor_like) >= max(1, int(0.8 * len(tensor_like))):
            return obj

    raise ValueError(" .pth  state_dict，。")


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def _make_state_dict_candidates(state_dict: Dict[str, torch.Tensor]):
    candidates = []
    sd0 = _strip_prefix_if_present(state_dict, "module.")
    candidates.append(("original/module-stripped", sd0))

    prefixes = ["gnn.", "model.", "encoder.", "module.gnn.", "module.model.", "module.encoder."]
    for prefix in prefixes:
        sd = _strip_prefix_if_present(state_dict, prefix)
        sd = _strip_prefix_if_present(sd, "module.")
        candidates.append((f"strip:{prefix}", sd))

    unique = []
    seen = set()
    for name, sd in candidates:
        key_tuple = tuple(sorted(sd.keys()))
        if key_tuple not in seen:
            unique.append((name, sd))
            seen.add(key_tuple)
    return unique


def _load_more_encoder_weights(model: nn.Module, weight_path: str | Path, device: torch.device) -> None:
    obj = _safe_torch_load(weight_path, map_location=device)
    raw_state_dict = _extract_state_dict(obj)

    best = None
    for name, sd in _make_state_dict_candidates(raw_state_dict):
        try:
            model.load_state_dict(sd, strict=True)
            print(f"[Online MORE] ：{name}，strict=True")
            return
        except RuntimeError:
            result = model.load_state_dict(sd, strict=False)
            missing_keys = list(result.missing_keys)
            unexpected_keys = list(result.unexpected_keys)
            score = len(missing_keys)
            if best is None or score < best[0]:
                best = (score, name, sd, missing_keys, unexpected_keys)

    if best is None:
        raise RuntimeError("[Online MORE] ： state_dict。")

    score, name, sd, missing_keys, unexpected_keys = best
    if len(missing_keys) > 0:
        msg = [
            "[Online MORE] ： GNN encoder 。",
            f"：{name}",
            f"missing_keys ：{len(missing_keys)}",
            " 20  missing_keys：",
            *[f"  {k}" for k in missing_keys[:20]],
        ]
        raise RuntimeError("\n".join(msg))

    model.load_state_dict(sd, strict=False)
    print(f"[Online MORE] ：{name}，strict=False")

    if len(unexpected_keys) > 0:
        print("[Online MORE] ， decoder。")
        for k in unexpected_keys[:20]:
            print("  ", k)
        if len(unexpected_keys) > 20:
            print(f"  ...  {len(unexpected_keys) - 20} ")


def _more_graph_dim(num_layer: int, emb_dim: int, jk: str) -> int:
    if str(jk).lower() == "concat":
        return int(num_layer + 1) * int(emb_dim)
    return int(emb_dim)


class OnlineMOREGraphEncoder(nn.Module):
    """
     MORE encoder： batch.more_x / more_edge_index / more_edge_attr  MORE.pth 。

     frozen=True， eval() + no_grad() + pooling 。
    """

    def __init__(
        self,
        weight_path: str | Path,
        num_layer: int = 5,
        emb_dim: int = 300,
        jk: str = "last",
        dropout_ratio: float = 0.0,
        gnn_type: str = "gin",
        pooling: str = "mean",
        frozen: bool = True,
        unfrozen_layers: Optional[int] = None,
    ):
        super().__init__()

        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f" MORE : {weight_path}")

        from model import GNN

        self.weight_path = str(weight_path)
        self.num_layer = int(num_layer)
        self.emb_dim = int(emb_dim)
        self.jk = str(jk)
        self.dropout_ratio = float(dropout_ratio)
        self.gnn_type = str(gnn_type)
        self.pooling = str(pooling).strip().lower()
        self.frozen = bool(frozen)
        self.requested_unfrozen_layers = unfrozen_layers

        if self.pooling not in {"mean", "sum", "max"}:
            raise ValueError("MORE_POOLING  mean / sum / max")

        self.more_encoder = GNN(
            num_layer=self.num_layer,
            emb_dim=self.emb_dim,
            JK=self.jk,
            drop_ratio=self.dropout_ratio,
            gnn_type=self.gnn_type,
        )
        _load_more_encoder_weights(self.more_encoder, self.weight_path, get_preferred_device())

        self.graph_dim = _more_graph_dim(self.num_layer, self.emb_dim, self.jk)
        self.unfrozen_layers = self._resolve_unfrozen_layers(unfrozen_layers)
        self._apply_trainable_policy()

    def _resolve_unfrozen_layers(self, unfrozen_layers: Optional[int]) -> int:
        if self.frozen:
            return 0
        if unfrozen_layers is None:
            return self.num_layer
        n = int(unfrozen_layers)
        if n < 0:
            raise ValueError(f"MORE_UNFROZEN_LAYERS  >= 0， {n}")
        if n > self.num_layer:
            raise ValueError(
                f"MORE_UNFROZEN_LAYERS={n}  MORE_NUM_LAYER={self.num_layer}"
            )
        return n

    @staticmethod
    def _set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for param in module.parameters():
            param.requires_grad = bool(requires_grad)

    def _apply_trainable_policy(self) -> None:
        self._set_module_requires_grad(self.more_encoder, False)

        if self.unfrozen_layers <= 0:
            self.more_encoder.eval()
            return

        start_layer = self.num_layer - self.unfrozen_layers

        
        if self.unfrozen_layers == self.num_layer:
            self._set_module_requires_grad(self.more_encoder, True)
            return

        for layer_idx in range(start_layer, self.num_layer):
            self._set_module_requires_grad(self.more_encoder.gnns[layer_idx], True)
            self._set_module_requires_grad(self.more_encoder.batch_norms[layer_idx], True)

    def _apply_training_mode_policy(self, mode: bool) -> None:
        if self.unfrozen_layers <= 0:
            self.more_encoder.eval()
            return

        if self.unfrozen_layers == self.num_layer:
            self.more_encoder.train(mode)
            return

        start_layer = self.num_layer - self.unfrozen_layers
        for layer_idx in range(0, start_layer):
            self.more_encoder.gnns[layer_idx].eval()
            self.more_encoder.batch_norms[layer_idx].eval()
        for layer_idx in range(start_layer, self.num_layer):
            self.more_encoder.gnns[layer_idx].train(mode)
            self.more_encoder.batch_norms[layer_idx].train(mode)

    def train(self, mode: bool = True):
        super().train(mode)
        
        self._apply_training_mode_policy(mode)
        return self

    def _pool(self, node_rep: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "sum":
            return global_add_pool(node_rep, batch)
        if self.pooling == "max":
            return global_max_pool(node_rep, batch)
        return global_mean_pool(node_rep, batch)

    def forward(self, data, return_node_embeddings: bool = False):
        if not hasattr(data, "more_x") or not hasattr(data, "more_edge_index") or not hasattr(data, "more_edge_attr"):
            raise ValueError(
                "batch  more_x / more_edge_index / more_edge_attr。"
                " data_loading.py， MORE.pth  MORE 。"
            )
        if not hasattr(data, "batch") or data.batch is None:
            raise ValueError(" MORE encoder  batch.batch。")

        device = next(self.more_encoder.parameters()).device
        x = data.more_x.to(device=device, dtype=torch.long, non_blocking=True)
        edge_index = data.more_edge_index.to(device=device, non_blocking=True)
        edge_attr = data.more_edge_attr.to(device=device, dtype=torch.long, non_blocking=True)
        batch = data.batch.to(device=device, non_blocking=True)

        if x.size(0) != batch.numel():
            raise ValueError(
                f"MORE  batch ：more_x={x.size(0)}, batch={batch.numel()}。"
            )

        if self.unfrozen_layers <= 0:
            with torch.no_grad():
                node_rep = self.more_encoder(x, edge_index, edge_attr)
        else:
            node_rep = self.more_encoder(x, edge_index, edge_attr)

        graph_vec = self._pool(node_rep, batch)

        if return_node_embeddings:
            return graph_vec, node_rep
        return graph_vec


class MORELayerNormAdapter(nn.Module):
    """LayerNorm adapter for online MORE graph vectors."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.graph_dim = int(in_dim)
        self.norm = nn.LayerNorm(self.in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"MORELayerNormAdapter  [B,D]， {tuple(x.shape)}")
        if x.size(1) != self.in_dim:
            raise ValueError(f"MORELayerNormAdapter ： D={self.in_dim}， D={x.size(1)}")
        return self.norm(x)


class GraphTransformerOnlineMOREAdapterGTConcatEncoder(nn.Module):
    """
     MORE  + adapter ， Graph Transformer  concat。

    ：
        mol_vec = cat(adapter_vec, GT_vec)

    return_node_embeddings=True ，interaction  Graph Transformer  atom tokens，
     MORE  node_rep， token-level self-attention 。
    """

    def __init__(
        self,
        graph_encoder: FlexibleTransformerEncoder,
        more_encoder: OnlineMOREGraphEncoder,
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.more_encoder = more_encoder
        self.more_adapter = MORELayerNormAdapter(in_dim=int(more_encoder.graph_dim))
        self.graph_dim = int(self.more_adapter.graph_dim) + int(self.graph_encoder.graph_dim)

    def forward(self, data, return_node_embeddings: bool = False):
        graph_vec, node_emb = self.graph_encoder(data=data, return_node_embeddings=True)
        more_vec = self.more_encoder(data)

        if more_vec.device != graph_vec.device or more_vec.dtype != graph_vec.dtype:
            more_vec = more_vec.to(device=graph_vec.device, dtype=graph_vec.dtype, non_blocking=True)

        adapter_vec = self.more_adapter(more_vec)
        out = torch.cat([adapter_vec, graph_vec], dim=-1)

        if return_node_embeddings:
            return out, node_emb
        return out





def _build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    enabled = bool(getattr(cfg, "LR_SCHEDULER_ENABLED", True))
    if not enabled:
        return None, None

    scheduler_name = str(getattr(cfg, "LR_SCHEDULER_NAME", "ReduceLROnPlateau")).strip().lower()
    monitor = str(getattr(cfg, "LR_SCHEDULER_MONITOR", "val_loss")).strip().lower()

    if scheduler_name in {"reducelronplateau", "plateau"}:
        if monitor == "val_r2":
            mode = str(getattr(cfg, "LR_SCHEDULER_MODE", "max")).strip().lower()
        else:
            mode = str(getattr(cfg, "LR_SCHEDULER_MODE", "min")).strip().lower()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(getattr(cfg, "LR_SCHEDULER_FACTOR", 0.5)),
            patience=int(getattr(cfg, "LR_SCHEDULER_PATIENCE", 10)),
            threshold=float(getattr(cfg, "LR_SCHEDULER_THRESHOLD", 1e-4)),
            threshold_mode=str(getattr(cfg, "LR_SCHEDULER_THRESHOLD_MODE", "rel")),
            cooldown=int(getattr(cfg, "LR_SCHEDULER_COOLDOWN", 0)),
            min_lr=float(getattr(cfg, "LR_SCHEDULER_MIN_LR", 1e-6)),
            eps=float(getattr(cfg, "LR_SCHEDULER_EPS", 1e-8)),
        )
        return scheduler, monitor

    raise ValueError(
        f" LR scheduler: {scheduler_name!r}。 'ReduceLROnPlateau'。"
    )





def _build_graph_transformer_encoder(sample_batch, cfg) -> tuple[FlexibleTransformerEncoder, int]:
    if not hasattr(sample_batch, "x"):
        raise ValueError("sample_batch  x，。")

    in_dim_mol = int(sample_batch.x.size(1))

    mol_encoder = FlexibleTransformerEncoder(
        in_dim=in_dim_mol,
        hidden_dims=cfg.GRAPH_TRANSFORMER_HIDDEN,
        num_layers=cfg.GRAPH_TRANSFORMER_LAYERS if isinstance(cfg.GRAPH_TRANSFORMER_HIDDEN, int) else None,
        out_dim=cfg.GRAPH_TRANSFORMER_OUT,
        heads=cfg.GRAPH_TRANSFORMER_HEADS,
        concat=True,
        edge_dim=cfg.GRAPH_TRANSFORMER_EDGE_DIM if getattr(cfg, "GRAPH_TRANSFORMER_USE_EDGE_ATTR", True) else None,
        dropout=cfg.GRAPH_TRANSFORMER_DROPOUT,
        activation="relu",
        use_bn=True,
        pooling="attention",
        beta=getattr(cfg, "GRAPH_TRANSFORMER_BETA", False),
        use_edge_attr=getattr(cfg, "GRAPH_TRANSFORMER_USE_EDGE_ATTR", True),
    ).to(cfg.DEVICE)

    readout_type = str(getattr(cfg, "GRAPH_TRANSFORMER_READOUT", "attention")).strip().lower()
    if readout_type in {"self_attention", "self-attention", "selfattention", "token_self_attention", "token-level-self-attention", "cls"}:
        mol_encoder = TokenSelfAttentionGraphReadoutEncoder(
            graph_encoder=mol_encoder,
            attention_heads=int(getattr(cfg, "GRAPH_TRANSFORMER_READOUT_HEADS", cfg.GRAPH_TRANSFORMER_HEADS)),
            attention_layers=int(getattr(cfg, "GRAPH_TRANSFORMER_READOUT_LAYERS", 1)),
            ffn_dim=getattr(cfg, "GRAPH_TRANSFORMER_READOUT_FFN_DIM", None),
            dropout=float(getattr(cfg, "GRAPH_TRANSFORMER_READOUT_DROPOUT", cfg.GRAPH_TRANSFORMER_DROPOUT)),
        ).to(cfg.DEVICE)
    elif readout_type not in {"attention", "mean"}:
        raise ValueError(
            f" GRAPH_TRANSFORMER_READOUT={readout_type!r}；"
            " 'attention' / 'mean' / 'self_attention'。"
        )

    return mol_encoder, in_dim_mol


def _normalize_solute_encoder_type(raw_type: str) -> str:
    key = str(raw_type).strip().lower()

    alias = {
        "transformer": "graph_transformer",
        "graphtransformer": "graph_transformer",
        "graph-transformer": "graph_transformer",

        
        "graph_transformer_more_online_ln_adapter_gt_concat": "graph_transformer_more_online_ln_adapter_gt_concat",
        "graph_transformer_online_more_ln_adapter_gt_concat": "graph_transformer_more_online_ln_adapter_gt_concat",
        "online_more_ln_adapter_gt_concat": "graph_transformer_more_online_ln_adapter_gt_concat",
        "more_online_ln_adapter_gt_concat": "graph_transformer_more_online_ln_adapter_gt_concat",
    }

    return alias.get(key, key)


def _build_mol_encoder(sample_batch, cfg):
    """
     SOLUTE_ENCODER_TYPE：

    1. graph_transformer
        Graph Transformer。

    2. graph_transformer_more_online_ln_adapter_gt_concat
       mol_vec = cat(adapter_vec_from_MORE_pth, GT_vec)。
    """

    raw_encoder_type = getattr(cfg, "SOLUTE_ENCODER_TYPE", "graph_transformer")
    encoder_type = _normalize_solute_encoder_type(raw_encoder_type)

    
    if encoder_type == "graph_transformer":
        graph_encoder, in_dim_mol = _build_graph_transformer_encoder(sample_batch, cfg)
        return graph_encoder, in_dim_mol

    
    
    if encoder_type == "graph_transformer_more_online_ln_adapter_gt_concat":
        graph_encoder, in_dim_mol = _build_graph_transformer_encoder(sample_batch, cfg)
        more_weight_path = getattr(cfg, "MORE_WEIGHT_PATH", None)
        if more_weight_path is None or str(more_weight_path).strip() == "":
            raise ValueError(
                " graph_transformer_more_online_ln_adapter_gt_concat ，"
                " MORE_WEIGHT_PATH， 'MORE.pth'。"
            )

        more_encoder = OnlineMOREGraphEncoder(
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

        mol_encoder = GraphTransformerOnlineMOREAdapterGTConcatEncoder(
            graph_encoder=graph_encoder,
            more_encoder=more_encoder,
        ).to(cfg.DEVICE)

        return mol_encoder, int(mol_encoder.graph_dim)

    raise ValueError(
        f" SOLUTE_ENCODER_TYPE: {raw_encoder_type!r}。"
        "：'graph_transformer'  'graph_transformer_more_online_ln_adapter_gt_concat'。"
    )





def _build_solvent_graph_encoder(sample_batch, cfg):
    in_dim_solvent = _infer_solvent_graph_in_dim(sample_batch)
    encoder_type = str(getattr(cfg, "SOLVENT_GRAPH_ENCODER_TYPE", "transformer")).strip().lower()

    if encoder_type == "transformer":
        sol_encoder = FlexibleTransformerEncoder(
            in_dim=in_dim_solvent,
            hidden_dims=cfg.SOLVENT_GRAPH_TRANSFORMER_HIDDEN,
            num_layers=cfg.SOLVENT_GRAPH_TRANSFORMER_LAYERS if isinstance(cfg.SOLVENT_GRAPH_TRANSFORMER_HIDDEN, int) else None,
            out_dim=cfg.SOLVENT_GRAPH_TRANSFORMER_OUT,
            heads=cfg.SOLVENT_GRAPH_TRANSFORMER_HEADS,
            concat=True,
            edge_dim=cfg.SOLVENT_GRAPH_TRANSFORMER_EDGE_DIM if getattr(cfg, "SOLVENT_GRAPH_TRANSFORMER_USE_EDGE_ATTR", True) else None,
            dropout=cfg.SOLVENT_GRAPH_TRANSFORMER_DROPOUT,
            activation="relu",
            use_bn=True,
            pooling="mean",
            beta=getattr(cfg, "SOLVENT_GRAPH_TRANSFORMER_BETA", False),
            use_edge_attr=getattr(cfg, "SOLVENT_GRAPH_TRANSFORMER_USE_EDGE_ATTR", True),
        ).to(cfg.DEVICE)
        return sol_encoder, sol_encoder.graph_dim, True

    if encoder_type == "gat":
        sol_encoder = FlexibleGATEncoder(
            in_dim=in_dim_solvent,
            hidden_dims=cfg.SOLVENT_GAT_HIDDEN,
            num_layers=cfg.SOLVENT_GAT_LAYERS if isinstance(cfg.SOLVENT_GAT_HIDDEN, int) else None,
            out_dim=cfg.SOLVENT_GAT_OUT,
            heads=cfg.SOLVENT_GAT_HEADS,
            concat=cfg.SOLVENT_GAT_CONCAT,
            dropout=cfg.SOLVENT_GAT_DROPOUT,
            activation="relu",
            use_bn=True,
            pooling="mean",
        ).to(cfg.DEVICE)
        return sol_encoder, sol_encoder.graph_dim, False

    raise ValueError(
        f" SOLVENT_GRAPH_ENCODER_TYPE: {encoder_type!r}。 'gat'  'transformer'。"
    )



def _infer_node_dim_from_encoder(encoder, encoder_name: str) -> int:
    """ encoder  node_emb ， encoder 。"""
    base = getattr(encoder, "graph_encoder", encoder)
    hidden_dims = getattr(base, "hidden_dims", None)
    if hidden_dims is None or len(hidden_dims) == 0:
        raise ValueError(
            f"{encoder_name}  token-level Self-Attention："
            " hidden_dims  node token 。"
        )
    return int(hidden_dims[-1])





def build_system(sample_batch, cfg) -> SimpleNamespace:
    mol_encoder, in_dim_mol = _build_mol_encoder(sample_batch, cfg)

    graph_encoder_type = getattr(cfg, "SOLVENT_GRAPH_ENCODER_TYPE", None)
    graph_encoder_type = None if graph_encoder_type is None else str(graph_encoder_type).strip().lower()

    use_solvent_graph_encoder = graph_encoder_type not in {None, "", "none", "null", "false"}

    if use_solvent_graph_encoder:
        sol_encoder, in_dim_sol, use_solvent_edge_attr = _build_solvent_graph_encoder(sample_batch, cfg)
    else:
        
        in_dim_sol = _infer_solvent_feat_dim(sample_batch)
        sol_encoder = nn.Identity().to(cfg.DEVICE)
        use_solvent_edge_attr = False

    raw_fusion_type = str(getattr(cfg, "FUSION_TYPE", "concat")).strip().lower()
    use_fusion_self_attention = raw_fusion_type in {
        "self_attention",
        "self-attention",
        "selfattention",
        "sa",
        "transformer",
    }

    if use_fusion_self_attention:
        if not use_solvent_graph_encoder:
            raise ValueError(
                "token-level Self-Attention  encoder。"
                " solvent_graph_encoder_type='transformer'。"
            )

        solute_token_dim = _infer_node_dim_from_encoder(mol_encoder, "mol_encoder")
        solvent_token_dim = _infer_node_dim_from_encoder(sol_encoder, "sol_encoder")

        fusion_encoder = SoluteSolventSelfAttentionFusion(
            solute_token_dim=solute_token_dim,
            solvent_token_dim=solvent_token_dim,
            attention_dim=int(getattr(cfg, "FUSION_ATTENTION_DIM", 128)),
            attention_heads=int(getattr(cfg, "FUSION_ATTENTION_HEADS", 4)),
            attention_layers=int(getattr(cfg, "FUSION_ATTENTION_LAYERS", 1)),
            ffn_dim=getattr(cfg, "FUSION_ATTENTION_FFN_DIM", None),
            dropout=float(getattr(cfg, "FUSION_ATTENTION_DROPOUT", 0.1)),
            output_mode=str(getattr(cfg, "FUSION_OUTPUT_MODE", "mean")),
        ).to(cfg.DEVICE)

        
        mlp_input_dim = int(mol_encoder.graph_dim) + int(in_dim_sol) + int(fusion_encoder.out_dim)
    else:
        if raw_fusion_type not in {"concat", "cat", "none", ""}:
            raise ValueError(
                f" FUSION_TYPE: {raw_fusion_type!r}。"
                " 'concat'  'self_attention'； Cross-Attention 。"
            )
        fusion_encoder = nn.Identity().to(cfg.DEVICE)
        mlp_input_dim = int(mol_encoder.graph_dim) + int(in_dim_sol)

    mlp = MultiTaskMLP(
        input_dim=mlp_input_dim,
        hidden_dim=cfg.MLP_HIDDEN,
        layers=cfg.MLP_LAYERS,
        dropout=cfg.MLP_DROPOUT,
        use_layernorm=True,
        out_dim=len(getattr(cfg, "TARGET_NAMES", ["abs", "emi", "plqy", "em"])),
        auto_to_device=False,
    ).to(cfg.DEVICE)

    modules_for_device = [mol_encoder, mlp]
    if use_solvent_graph_encoder:
        modules_for_device.append(sol_encoder)
    if use_fusion_self_attention:
        modules_for_device.append(fusion_encoder)

    
    modules_for_device = [m.to(cfg.DEVICE) for m in modules_for_device]
    mol_encoder = modules_for_device[0]
    mlp = modules_for_device[1]
    offset = 2
    if use_solvent_graph_encoder:
        sol_encoder = modules_for_device[offset]
        offset += 1
    if use_fusion_self_attention:
        fusion_encoder = modules_for_device[offset]

    device = ensure_same_device(*modules_for_device)

    all_trainable_params = list(mol_encoder.parameters()) + list(mlp.parameters())
    if use_solvent_graph_encoder:
        all_trainable_params += list(sol_encoder.parameters())
    if use_fusion_self_attention:
        all_trainable_params += list(fusion_encoder.parameters())

    optimizer = torch.optim.AdamW(
        all_trainable_params,
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler, scheduler_monitor = _build_scheduler(optimizer, cfg)

    early_stopper = EarlyStopping(
        patience=cfg.EARLY_STOPPING_PATIENCE,
        min_delta=cfg.EARLY_STOPPING_MIN_DELTA,
        mode=cfg.EARLY_STOPPING_MODE,
        save_path=None,
        verbose=cfg.EARLY_STOPPING_VERBOSE,
    )

    return SimpleNamespace(
        mol_encoder=mol_encoder,
        sol_encoder=sol_encoder,
        fusion_encoder=fusion_encoder,
        optimizer=optimizer,
        mlp=mlp,
        early_stopper=early_stopper,
        scheduler=scheduler,
        scheduler_monitor=scheduler_monitor,
        all_trainable_params=all_trainable_params,
        device=device,
        in_dim_mol=in_dim_mol,
        in_dim_sol=in_dim_sol,
        solvent_feat_dim=in_dim_sol,
        mlp_input_dim=mlp_input_dim,
        fusion_type=raw_fusion_type,
        use_fusion_self_attention=use_fusion_self_attention,
        fusion_output_dim=int(getattr(fusion_encoder, "out_dim", 0)) if use_fusion_self_attention else 0,
        use_solvent_graph_encoder=use_solvent_graph_encoder,
        use_solvent_edge_attr=use_solvent_edge_attr,
    )

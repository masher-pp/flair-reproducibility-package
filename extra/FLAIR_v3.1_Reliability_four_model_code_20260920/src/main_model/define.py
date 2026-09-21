from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, TransformerConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax


_ALLOWED_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
}


def _normalize_hidden_dims(hidden_dims, num_layers: Optional[int]) -> list[int]:
    if isinstance(hidden_dims, int):
        if num_layers is None:
            raise ValueError(" hidden_dims  int ， num_layers")
        if num_layers < 1:
            raise ValueError("num_layers  >= 1")
        return [int(hidden_dims)] * int(num_layers)

    if isinstance(hidden_dims, (list, tuple)):
        hidden_dims = [int(x) for x in hidden_dims]
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims ")
        return hidden_dims

    raise TypeError("hidden_dims  int  list/tuple[int]")


def _normalize_heads(heads, num_layers: int) -> list[int]:
    if isinstance(heads, int):
        heads = [int(heads)] * num_layers
    elif isinstance(heads, (list, tuple)):
        heads = [int(x) for x in heads]
        if len(heads) != num_layers:
            raise ValueError(" heads  list/tuple，")
    else:
        raise TypeError("heads  int  list/tuple[int]")

    if any(h < 1 for h in heads):
        raise ValueError("heads  >= 1")
    return heads


def _build_activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key not in _ALLOWED_ACTIVATIONS:
        raise ValueError("activation  'relu', 'gelu', 'elu'")
    return _ALLOWED_ACTIVATIONS[key]()


def _edge_attr_to_gcn_weight(edge_attr: torch.Tensor | None) -> torch.Tensor | None:
    if edge_attr is None:
        return None
    if not isinstance(edge_attr, torch.Tensor):
        edge_attr = torch.as_tensor(edge_attr)
    if edge_attr.numel() == 0:
        return None
    if edge_attr.ndim == 1:
        weight = edge_attr.float()
    else:
        weight = edge_attr.float().mean(dim=-1)
    weight = torch.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)
    return weight.clamp(min=0.0)


class _BaseGraphEncoder(nn.Module):
    """。 mean / attention  pooling。"""

    def __init__(self, dropout: float = 0.0, activation: str = "relu", use_bn: bool = True, pooling: str = "mean"):
        super().__init__()
        pooling = str(pooling).lower()
        if pooling not in {"mean", "attention"}:
            raise ValueError(" 'mean'  'attention' pooling")
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)
        self.pooling = pooling
        self.act = _build_activation(activation)
        self.attn_gate: nn.Module | None = None

    @staticmethod
    def _unpack_inputs(data=None, x=None, edge_index=None, batch=None, edge_attr=None):
        if data is not None:
            x = data.x
            edge_index = data.edge_index
            batch = getattr(data, "batch", None)
            if edge_attr is None:
                edge_attr = getattr(data, "edge_attr", None)

        if x is None or edge_index is None:
            raise ValueError(" data， x  edge_index")

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        return x, edge_index, batch, edge_attr

    @staticmethod
    def _mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return global_mean_pool(x, batch)

    def _attention_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.attn_gate is None:
            raise ValueError(" encoder  attention pooling  gate 。")

        gate_logits = self.attn_gate(x).view(-1)
        gate_alpha = softmax(gate_logits, batch)
        return global_add_pool(x * gate_alpha.unsqueeze(-1), batch)

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "attention":
            return self._attention_pool(x, batch)
        return self._mean_pool(x, batch)


class FlexibleGCNEncoder(_BaseGraphEncoder):
    """ GCN ， global_mean_pool。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dims,
        num_layers: int | None = None,
        out_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
        use_bn: bool = True,
        pooling: str = "mean",
    ):
        super().__init__(dropout=dropout, activation=activation, use_bn=use_bn, pooling=pooling)

        hidden_dims = _normalize_hidden_dims(hidden_dims, num_layers)
        dims = [int(in_dim)] + hidden_dims

        self.in_dim = int(in_dim)
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(self.num_layers)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(dims[i + 1]) for i in range(self.num_layers)]
        ) if self.use_bn else nn.ModuleList()

        final_dim = hidden_dims[-1]
        self.proj = nn.Linear(final_dim, out_dim) if out_dim is not None else None
        self.graph_dim = int(out_dim) if out_dim is not None else final_dim

    def forward(
        self,
        data=None,
        x=None,
        edge_index=None,
        batch=None,
        edge_attr=None,
        return_node_embeddings: bool = False,
    ):
        x, edge_index, batch, edge_attr = self._unpack_inputs(
            data=data,
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
        )
        edge_weight = _edge_attr_to_gcn_weight(edge_attr)

        for i, conv in enumerate(self.convs):
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_weight=edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.act(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        node_emb = x
        graph_emb = self._pool(node_emb, batch)
        if self.proj is not None:
            graph_emb = self.proj(graph_emb)

        if return_node_embeddings:
            return graph_emb, node_emb
        return graph_emb


class FlexibleGATEncoder(_BaseGraphEncoder):
    """ GAT ， global_mean_pool。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dims,
        num_layers: int | None = None,
        out_dim: int | None = None,
        heads=4,
        concat: bool = True,
        dropout: float = 0.0,
        activation: str = "relu",
        use_bn: bool = True,
        pooling: str = "mean",
    ):
        super().__init__(dropout=dropout, activation=activation, use_bn=use_bn, pooling=pooling)

        hidden_dims = _normalize_hidden_dims(hidden_dims, num_layers)
        heads = _normalize_heads(heads, len(hidden_dims))

        self.in_dim = int(in_dim)
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.concat = bool(concat)
        self.heads = heads

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        prev_dim = int(in_dim)
        for layer_idx, (hdim, n_heads) in enumerate(zip(hidden_dims, heads), start=1):
            if self.concat:
                if hdim % n_heads != 0:
                    raise ValueError(
                        f"{layer_idx} hidden_dims={hdim}  heads={n_heads} ；"
                        " concat=True ， = out_channels * heads"
                    )
                out_channels = hdim // n_heads
            else:
                out_channels = hdim

            self.convs.append(
                GATConv(
                    in_channels=prev_dim,
                    out_channels=out_channels,
                    heads=n_heads,
                    concat=self.concat,
                    dropout=self.dropout,
                )
            )
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(hdim))
            prev_dim = hdim

        self.proj = nn.Linear(prev_dim, out_dim) if out_dim is not None else None
        self.graph_dim = int(out_dim) if out_dim is not None else prev_dim

        if self.pooling == "attention":
            self.attn_gate = nn.Linear(prev_dim, 1)

    def forward(
        self,
        data=None,
        x=None,
        edge_index=None,
        batch=None,
        edge_attr=None,
        return_node_embeddings: bool = False,
    ):
        x, edge_index, batch, edge_attr = self._unpack_inputs(
            data=data,
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
        )
        edge_weight = _edge_attr_to_gcn_weight(edge_attr)

        for i, conv in enumerate(self.convs):
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_weight=edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.act(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        node_emb = x
        graph_emb = self._pool(node_emb, batch)
        if self.proj is not None:
            graph_emb = self.proj(graph_emb)

        if return_node_embeddings:
            return graph_emb, node_emb
        return graph_emb


class FlexibleTransformerEncoder(_BaseGraphEncoder):
    """ Graph Transformer ， global_mean_pool。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dims,
        num_layers: int | None = None,
        out_dim: int | None = None,
        heads=4,
        concat: bool = True,
        edge_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
        use_bn: bool = True,
        pooling: str = "mean",
        beta: bool = False,
        use_edge_attr: bool = True,
    ):
        super().__init__(dropout=dropout, activation=activation, use_bn=use_bn, pooling=pooling)

        hidden_dims = _normalize_hidden_dims(hidden_dims, num_layers)
        heads = _normalize_heads(heads, len(hidden_dims))

        self.in_dim = int(in_dim)
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.concat = bool(concat)
        self.edge_dim = edge_dim
        self.beta = bool(beta)
        self.use_edge_attr = bool(use_edge_attr)
        self.heads = heads

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        prev_dim = int(in_dim)
        for layer_idx, (hdim, n_heads) in enumerate(zip(hidden_dims, heads), start=1):
            if self.concat:
                if hdim % n_heads != 0:
                    raise ValueError(
                        f"{layer_idx} hidden_dims={hdim}  heads={n_heads} ；"
                        " concat=True ， = out_channels * heads"
                    )
                out_channels = hdim // n_heads
            else:
                out_channels = hdim

            self.convs.append(
                TransformerConv(
                    in_channels=prev_dim,
                    out_channels=out_channels,
                    heads=n_heads,
                    concat=self.concat,
                    dropout=self.dropout,
                    edge_dim=edge_dim,
                    beta=self.beta,
                )
            )
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(hdim))
            prev_dim = hdim

        self.proj = nn.Linear(prev_dim, out_dim) if out_dim is not None else None
        self.graph_dim = int(out_dim) if out_dim is not None else prev_dim

        if self.pooling == "attention":
            self.attn_gate = nn.Linear(prev_dim, 1)

    def forward(
        self,
        data=None,
        x=None,
        edge_index=None,
        batch=None,
        edge_attr=None,
        return_node_embeddings: bool = False,
    ):
        x, edge_index, batch, edge_attr = self._unpack_inputs(
            data=data,
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
        )

        for i, conv in enumerate(self.convs):
            if self.use_edge_attr and edge_attr is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            if self.use_bn:
                x = self.bns[i](x)
            x = self.act(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        node_emb = x
        graph_emb = self._pool(node_emb, batch)
        if self.proj is not None:
            graph_emb = self.proj(graph_emb)

        if return_node_embeddings:
            return graph_emb, node_emb
        return graph_emb


__all__ = [
    "FlexibleGCNEncoder",
    "FlexibleGATEncoder",
    "FlexibleTransformerEncoder",
]

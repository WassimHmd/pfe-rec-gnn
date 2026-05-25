"""HGT layer + encoder with Relative Temporal Encoding (Hu et al. 2020 §3.4).

Per-edge (s -> t) along relation φ(e):

    H_tilde[s, t] = H[s] + RTE(T(t) - T(s))
    K[e] = K-Linear_{τ(s)}(H_tilde[s, t]) · split per head · W^ATT_{φ(e)}
    V[e] = V-Linear_{τ(s)}(H_tilde[s, t]) · split per head · W^MSG_{φ(e)}
    Q[e] = Q-Linear_{τ(t)}(H[t])         · split per head
    score[e, h] = (Q[e, h] · K[e, h]) / √(d/H) · μ_{φ(e), h}
    attn[e, h]  = softmax_{e | dst(e)=t}(score[e, h])
    msg[e, h]   = V[e, h] · attn[e, h]
    out[t, h]   = Σ_{e | dst(e)=t} msg[e, h]
    out[t]      = out-Linear_{τ(t)}(concat_h out[t, h])

Linearity trick to avoid recomputing per-edge bulk projections:
    K[e] = K_base[src(e)] + K_lin_{τ(src)}(RTE(Δt))
    V[e] = V_base[src(e)] + V_lin_{τ(src)}(RTE(Δt))
where K_base, V_base are per-node bulk projections computed once.

Residual is added by the encoder wrapper, matching HGTEncoder's pattern.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.utils import softmax as pyg_softmax

from .base import Encoder

NodeType = str
EdgeType = Tuple[str, str, str]


def sinusoidal_time(t: Tensor, dim: int) -> Tensor:
    """Transformer-style positional encoding of a continuous time value.

    Args:
        t:   (N,) float tensor of timestamps (any continuous units; we use years-since-1900).
        dim: output dim (must be even).
    Returns: (N, dim) float tensor.
    """
    assert dim % 2 == 0, f"rte_dim must be even, got {dim}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = t.float().unsqueeze(-1) * freqs.unsqueeze(0)   # (N, half)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class HGTRTEConv(nn.Module):
    """Single HGT layer with RTE injection.

    in_channels and out_channels must be equal (matches PyG HGTConv pattern for
    stackable layers).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        metadata: Tuple[List[NodeType], List[EdgeType]],
        num_heads: int,
        rte_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert in_channels == out_channels, \
            f"HGTRTEConv requires in_channels == out_channels for residual; got {in_channels} != {out_channels}"
        assert out_channels % num_heads == 0, \
            f"out_channels ({out_channels}) must be divisible by num_heads ({num_heads})"

        node_types, edge_types = metadata
        self.node_types: List[NodeType] = list(node_types)
        self.edge_types: List[EdgeType] = list(edge_types)
        self._edge_type_to_idx = {et: i for i, et in enumerate(self.edge_types)}

        self.num_heads = num_heads
        self.head_dim  = out_channels // num_heads
        self.out_channels = out_channels
        self.rte_dim = rte_dim if rte_dim is not None else out_channels
        assert self.rte_dim % 2 == 0

        # Per-node-type K, Q, V, out projections (regular Linear is more portable than PyG HeteroDictLinear).
        self.k_lin   = nn.ModuleDict({nt: nn.Linear(in_channels, out_channels) for nt in self.node_types})
        self.q_lin   = nn.ModuleDict({nt: nn.Linear(in_channels, out_channels) for nt in self.node_types})
        self.v_lin   = nn.ModuleDict({nt: nn.Linear(in_channels, out_channels) for nt in self.node_types})
        self.out_lin = nn.ModuleDict({nt: nn.Linear(out_channels, out_channels) for nt in self.node_types})

        # Per-node-type RTE projection — applied to RTE(Δt) for messages where this type is the SOURCE.
        # Bias=False so the encoded "zero time delta" maps to true zero (no spurious offset on Δt=0 edges).
        self.rte_to_k = nn.ModuleDict({nt: nn.Linear(self.rte_dim, out_channels, bias=False) for nt in self.node_types})
        self.rte_to_v = nn.ModuleDict({nt: nn.Linear(self.rte_dim, out_channels, bias=False) for nt in self.node_types})

        # Per-edge-type W^ATT, W^MSG: per-head head_dim × head_dim transformations.
        # Use a flat parameter name ("__".join(et)) so they're addressable by ModuleDict-safe string keys.
        self.w_att = nn.ParameterDict()
        self.w_msg = nn.ParameterDict()
        for et in self.edge_types:
            key = self._et_key(et)
            self.w_att[key] = nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim))
            self.w_msg[key] = nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim))
            nn.init.xavier_uniform_(self.w_att[key])
            nn.init.xavier_uniform_(self.w_msg[key])

        # Per-edge-type, per-head μ scaling (Hu et al. eq. 4).
        self.mu = nn.Parameter(torch.ones(len(self.edge_types), num_heads))

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _et_key(et: EdgeType) -> str:
        # ParameterDict keys must be strings; tuples aren't hashable as keys directly.
        return "__".join(et)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        t_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        # 1. Bulk per-node K, Q, V projections.
        #    Reshape to (N, num_heads, head_dim) so per-head ops are clean.
        k_base = {nt: self.k_lin[nt](x).view(-1, self.num_heads, self.head_dim) for nt, x in x_dict.items()}
        q_base = {nt: self.q_lin[nt](x).view(-1, self.num_heads, self.head_dim) for nt, x in x_dict.items()}
        v_base = {nt: self.v_lin[nt](x).view(-1, self.num_heads, self.head_dim) for nt, x in x_dict.items()}

        # 2. Accumulators per destination type.
        out_dict: Dict[NodeType, Tensor] = {
            nt: torch.zeros(
                (x_dict[nt].size(0), self.num_heads, self.head_dim),
                device=x_dict[nt].device, dtype=x_dict[nt].dtype,
            )
            for nt in x_dict
        }

        # 3. Per-edge-type message passing.
        for et, edge_index in edge_index_dict.items():
            if edge_index.numel() == 0 or et not in self._edge_type_to_idx:
                continue
            src_type, _, dst_type = et
            if src_type not in x_dict or dst_type not in x_dict:
                continue   # subsampled batch may not contain this type pair

            src_idx, dst_idx = edge_index[0], edge_index[1]
            E = src_idx.size(0)
            key = self._et_key(et)

            # 3a. Per-edge K and V with RTE residual: K = K_base[src] + K_rte(Δt)
            t_src = t_dict[src_type][src_idx] if src_type in t_dict else torch.zeros(E, device=src_idx.device)
            t_dst = t_dict[dst_type][dst_idx] if dst_type in t_dict else torch.zeros(E, device=src_idx.device)
            dt = t_dst - t_src                                                # (E,)
            rte = sinusoidal_time(dt, self.rte_dim)                           # (E, rte_dim)
            rte_k = self.rte_to_k[src_type](rte).view(E, self.num_heads, self.head_dim)
            rte_v = self.rte_to_v[src_type](rte).view(E, self.num_heads, self.head_dim)

            k_e = k_base[src_type][src_idx] + rte_k                           # (E, H, d_h)
            v_e = v_base[src_type][src_idx] + rte_v
            q_e = q_base[dst_type][dst_idx]                                   # (E, H, d_h)

            # 3b. Per-relation per-head W^ATT and W^MSG transformations:
            #     k_e[e, h, :] ← k_e[e, h, :] @ w_att[key, h, :, :]
            k_e = torch.einsum("ehd,hdf->ehf", k_e, self.w_att[key])
            v_e = torch.einsum("ehd,hdf->ehf", v_e, self.w_msg[key])

            # 3c. Per-edge attention scores, scaled by √d_h and the edge-type μ.
            score = (q_e * k_e).sum(dim=-1)                                   # (E, H)
            score = score / (self.head_dim ** 0.5)
            score = score * self.mu[self._edge_type_to_idx[et]].unsqueeze(0)  # (E, H) ← broadcast over edges

            # 3d. Softmax over edges sharing the same target (dst_idx).
            attn = pyg_softmax(score, dst_idx)                                # (E, H)
            attn = self.dropout(attn)

            # 3e. Aggregate: msg = V * attn (per head), scatter-sum to target.
            msg = v_e * attn.unsqueeze(-1)                                    # (E, H, d_h)
            out_dict[dst_type].index_add_(0, dst_idx, msg)

        # 4. Per-type output projection.
        result: Dict[NodeType, Tensor] = {}
        for nt, out in out_dict.items():
            flat = out.reshape(-1, self.out_channels)
            result[nt] = self.out_lin[nt](flat)
        return result


class HGTRTEEncoder(Encoder):
    """Stacked HGTRTEConv with residual + dropout. Forward signature accepts t_dict."""

    def __init__(
        self,
        metadata: Tuple[List[NodeType], List[EdgeType]],
        in_dims: Dict[NodeType, int],
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        rte_dim: int | None = None,
    ):
        super().__init__()
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_dim, d_model) for nt, in_dim in in_dims.items()
        })
        self.convs = nn.ModuleList([
            HGTRTEConv(d_model, d_model, metadata, num_heads, rte_dim=rte_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        t_dict: Dict[NodeType, Tensor] | None = None,
    ) -> Dict[NodeType, Tensor]:
        if t_dict is None:
            raise RuntimeError(
                "HGTRTEEncoder requires t_dict (per-node-type timestamps). "
                "Re-run preprocess.ipynb step 10 so data[nt].t exists for every node type."
            )

        h_dict = {nt: self.input_proj[nt](x) for nt, x in x_dict.items()}
        for conv in self.convs:
            h_new = conv(h_dict, edge_index_dict, t_dict)
            h_dict = {
                nt: self.dropout(h_new[nt]) + h_dict[nt] if nt in h_new else h_dict[nt]
                for nt in h_dict
            }
        return h_dict

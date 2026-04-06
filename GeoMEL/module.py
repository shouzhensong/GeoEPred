"""
Neural network modules for GeoMEL.

Includes:
    - GVP:  Geometric Vector Perceptron
    - GVPConv:  GVP message-passing convolution
    - GVPEncoder:  Multi-layer GVP-GNN for structure encoding
    - AttentionPooling:  Learnable attention-weighted aggregation
    - SeqEncoderV6:  ESM feature encoder with Transformer + Attention Pooling
    - TransformerClassifierV6:  Feature Tokenization + Transformer classifier head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


# ── GVP: Geometric Vector Perceptron ─────────────────────────────────────────


class GVP(nn.Module):
    """Geometric Vector Perceptron.

    Processes pairs of scalar and vector features while maintaining
    SO(3) equivariance on the vector channel.

    Args:
        in_dims:  ``(scalar_in, vector_in)``
        out_dims: ``(scalar_out, vector_out)``
        act:      Tuple of activations ``(scalar_act, vector_gate_act)``.
    """

    def __init__(self, in_dims, out_dims, act=(F.relu, torch.sigmoid)):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.act_s, self.act_v = act

        self.Wh = (
            nn.Linear(self.vi, self.vo, bias=False) if self.vi > 0 else None
        )
        self.Ws = nn.Linear(
            self.si + (self.vo if self.vi > 0 else 0), self.so
        )

    def forward(self, x):
        s, v = x
        if self.Wh and v is not None:
            v = self.Wh(v.transpose(-1, -2)).transpose(-1, -2)
            s = torch.cat([s, torch.norm(v, dim=-1)], -1)
        s = self.Ws(s)
        if self.act_s:
            s = self.act_s(s)
        if self.vo > 0 and v is not None and self.act_v:
            v = v * self.act_v(torch.norm(v, dim=-1, keepdim=True))
        else:
            v = None
        return s, v


# ── GVPConv: Message-passing layer ───────────────────────────────────────────


class GVPConv(nn.Module):
    """GVP-based message-passing convolution.

    Concatenates source, destination, and edge features, then applies a
    GVP to compute messages which are aggregated via mean pooling.

    Args:
        in_dims:   Node feature dimensions ``(scalar, vector)``.
        out_dims:  Output feature dimensions ``(scalar, vector)``.
        edge_dims: Edge feature dimensions ``(scalar, vector)``.
    """

    def __init__(self, in_dims, out_dims, edge_dims):
        super().__init__()
        self.msg = GVP(
            (2 * in_dims[0] + edge_dims[0], 2 * in_dims[1] + edge_dims[1]),
            out_dims,
        )

    def forward(self, x, edge_index, edge_attr):
        s, v = x
        es, ev = edge_attr
        src, dst = edge_index

        ms = torch.cat([s[src], s[dst], es], -1)
        mv = torch.cat([v[src], v[dst], ev], 1) if v is not None else ev
        ms, mv = self.msg((ms, mv))

        out_s = scatter_mean(ms, dst, dim=0, dim_size=s.size(0))
        out_v = (
            scatter_mean(mv, dst, dim=0, dim_size=s.size(0))
            if mv is not None
            else None
        )
        return out_s, out_v


# ── GVPEncoder: Structure branch ─────────────────────────────────────────────


class GVPEncoder(nn.Module):
    """Multi-layer GVP-GNN encoder for protein structure.

    Embeds node and edge geometric features, then applies multiple layers
    of GVP message-passing with residual connections.

    Args:
        node_dims:  Input node dims ``(scalar, vector)``.  Default ``(6, 3)``.
        edge_dims:  Input edge dims ``(scalar, vector)``.  Default ``(32, 1)``.
        hidden:     Hidden dims ``(scalar, vector)``.      Default ``(100, 16)``.
        layers:     Number of GVP convolution layers.      Default ``3``.
    """

    def __init__(
        self,
        node_dims=(6, 3),
        edge_dims=(32, 1),
        hidden=(100, 16),
        layers=3,
    ):
        super().__init__()
        self.node_emb = GVP(node_dims, hidden)
        self.edge_emb = GVP(edge_dims, edge_dims)
        self.convs = nn.ModuleList(
            [GVPConv(hidden, hidden, edge_dims) for _ in range(layers)]
        )
        self.out = nn.Linear(hidden[0], hidden[0])

    def forward(self, batch):
        h_s, h_v = self.node_emb((batch.node_s, batch.node_v))
        e_s, e_v = self.edge_emb((batch.edge_s, batch.edge_v))

        for conv in self.convs:
            ds, dv = conv((h_s, h_v), batch.edge_index, (e_s, e_v))
            h_s = h_s + ds
            h_v = (h_v + dv) if (h_v is not None and dv is not None) else h_v

        graph_repr = self.out(scatter_mean(h_s, batch.batch, dim=0))
        return graph_repr, h_s


# ── Attention Pooling ─────────────────────────────────────────────────────────


class AttentionPooling(nn.Module):
    """Learnable attention pooling over a sequence of token embeddings.

    Computes importance weights for each position and returns the
    weighted sum, allowing the model to focus on key residues (e.g.
    N-terminal signal peptides in T3SEs).

    Args:
        dim: Embedding dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x:    ``(B, L, D)`` token embeddings.
            mask: ``(B, L)`` boolean mask, ``True`` = padding.

        Returns:
            ``(B, D)`` pooled representation.
        """
        scores = self.attn(x).squeeze(-1)  # (B, L)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)
        weights = F.softmax(scores, dim=-1)  # (B, L)
        return (weights.unsqueeze(-1) * x).sum(1)  # (B, D)


# ── SeqEncoderV6: Sequence branch ────────────────────────────────────────────


class SeqEncoderV6(nn.Module):
    """Sequence encoder.

    Projects ESM-2 residue-level embeddings to a lower dimension, refines
    them with a 2-layer Transformer encoder, then aggregates via Attention
    Pooling.

    Args:
        in_dim:  Input dimension (ESM embedding size).
        hid:     Hidden / output dimension.
        layers:  Number of Transformer encoder layers.
    """

    def __init__(self, in_dim: int = 2560, hid: int = 256, layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid)
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hid, 8, hid * 4, 0.1, batch_first=True
            ),
            layers,
        )
        self.pool = AttentionPooling(hid)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        """
        Args:
            x:    ``(B, L, in_dim)`` padded ESM embeddings.
            lens: ``(B,)`` valid lengths.

        Returns:
            ``(global_repr, token_repr)`` — both ``torch.Tensor``.
        """
        x = self.proj(x)
        B, L, _ = x.shape
        mask = torch.arange(L, device=x.device).expand(B, L) >= lens.unsqueeze(1)
        x = self.tf(x, src_key_padding_mask=mask)
        global_repr = self.pool(x, mask)
        return global_repr, x


# ── TransformerClassifierV6: Feature Tokenization + Classification ────────────


class TransformerClassifierV6(nn.Module):
    """Transformer classifier head with Feature Tokenization.

    Converts the fused representation into multiple semantic tokens:
        - ``[CLS]``:    Learnable classification token
        - ``seq``:      Sequence sub-space
        - ``struct``:   Structure sub-space
        - ``cross``:    Sequence × Structure interaction (Hadamard product)

    A Transformer encoder then learns cross-token interactions before
    the ``[CLS]`` output is projected to class logits.

    Args:
        seq_dim:     Dimension of the sequence feature.
        struct_dim:  Dimension of the structure feature.
        d_model:     Transformer hidden dimension.
        n_heads:     Number of attention heads.
        n_layers:    Number of Transformer layers.
        n_cls:       Number of output classes.
        dropout:     Dropout rate.
    """

    def __init__(
        self,
        seq_dim: int,
        struct_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_cls: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Feature tokenizers
        self.seq_tokenizer = nn.Sequential(
            nn.LayerNorm(seq_dim), nn.Linear(seq_dim, d_model)
        )
        self.struct_tokenizer = nn.Sequential(
            nn.LayerNorm(struct_dim), nn.Linear(struct_dim, d_model)
        )

        # Cross-interaction token (low-rank bilinear)
        self.cross_proj_seq = nn.Linear(seq_dim, d_model)
        self.cross_proj_struct = nn.Linear(struct_dim, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional embedding (CLS + seq + struct + cross = 4 positions)
        self.pos_embed = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            d_model * 4,
            dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_cls),
        )

    def forward(
        self, seq_feat: torch.Tensor, struct_feat: torch.Tensor
    ) -> torch.Tensor:
        B = seq_feat.size(0)

        tok_seq = self.seq_tokenizer(seq_feat).unsqueeze(1)
        tok_struct = self.struct_tokenizer(struct_feat).unsqueeze(1)

        cross = torch.tanh(self.cross_proj_seq(seq_feat)) * torch.tanh(
            self.cross_proj_struct(struct_feat)
        )
        tok_cross = cross.unsqueeze(1)

        cls = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat([cls, tok_seq, tok_struct, tok_cross], dim=1)
        tokens = tokens + self.pos_embed

        tokens = self.transformer(tokens)

        cls_out = tokens[:, 0]
        return self.head(cls_out)

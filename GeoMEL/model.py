"""
GeoMEL — Main model definition.

Dual-branch architecture:
    - Sequence branch:   ESM-2 → SeqEncoderV6 (Attention Pooling) → 256d
    - Structure branch:  PDB → GVPEncoder (3-layer GVP-GNN) → 100d
    - Fusion:            LayerNorm alignment → Feature Tokenization
    - Classifier:        Transformer ([CLS] + semantic tokens) → 6 classes
    - Auxiliary:         NT-Xent contrastive projection heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import GVPEncoder, SeqEncoderV6, TransformerClassifierV6


class GeoMEL(nn.Module):
    """GeoMEL: Geometric Multi-modal Effector Learner.

    Args:
        n_cls:         Number of output classes.           Default ``6``.
        esm_dim:       ESM-2 embedding dimension.          Default ``2560``.
        seq_hid:       Sequence branch hidden dimension.   Default ``256``.
        struct_hid:    Structure branch hidden dimension.  Default ``100``.
        cls_d_model:   Transformer classifier dimension.   Default ``128``.
        cls_n_heads:   Number of attention heads.          Default ``4``.
        cls_n_layers:  Number of Transformer layers.       Default ``2``.
    """

    def __init__(
        self,
        n_cls: int = 6,
        esm_dim: int = 2560,
        seq_hid: int = 256,
        struct_hid: int = 100,
        cls_d_model: int = 128,
        cls_n_heads: int = 4,
        cls_n_layers: int = 2,
    ):
        super().__init__()

        # ── Encoders ──
        self.seq_enc = SeqEncoderV6(esm_dim, seq_hid)
        self.struct_enc = GVPEncoder(hidden=(struct_hid, 16))

        # ── Feature normalisation (align modality distributions) ──
        self.seq_norm = nn.LayerNorm(seq_hid)
        self.struct_norm = nn.LayerNorm(struct_hid)

        # ── Transformer classifier with Feature Tokenization ──
        self.classifier = TransformerClassifierV6(
            seq_hid, struct_hid, cls_d_model, cls_n_heads, cls_n_layers, n_cls
        )

        # ── Contrastive projection heads ──
        proj_dim = 128
        self.seq_projector = nn.Sequential(
            nn.Linear(seq_hid, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.struct_projector = nn.Sequential(
            nn.Linear(struct_hid, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            batch: Dictionary containing:
                - ``esm_feature``:  ``(B, L, esm_dim)`` padded ESM embeddings.
                - ``valid_lens``:   ``(B,)`` valid sequence lengths.
                - ``graph_data``:   ``torch_geometric.data.Batch`` of PDB graphs.

        Returns:
            ``(logits, seq_proj, struct_proj)``
                - ``logits``:      ``(B, n_cls)`` classification logits.
                - ``seq_proj``:    ``(B, proj_dim)`` L2-normalised sequence embedding.
                - ``struct_proj``: ``(B, proj_dim)`` L2-normalised structure embedding.
        """
        # Encode
        seq_global, _ = self.seq_enc(batch["esm_feature"], batch["valid_lens"])
        struct_global, _ = self.struct_enc(batch["graph_data"])

        # Normalise
        seq_global = self.seq_norm(seq_global)
        struct_global = self.struct_norm(struct_global)

        # Classify
        logits = self.classifier(seq_global, struct_global)

        # Contrastive projections
        seq_proj = F.normalize(self.seq_projector(seq_global), dim=-1)
        struct_proj = F.normalize(self.struct_projector(struct_global), dim=-1)

        return logits, seq_proj, struct_proj

"""
GeoMEL: Geometric Multi-modal Effector Learner

A dual-branch deep learning model for bacterial effector protein classification,
integrating ESM-2 sequence embeddings with GVP-GNN structural features via
Feature Tokenization and Transformer-based fusion.

Architecture:
    - Sequence Branch:  ESM-2 (36L, 3B) → Attention Pooling → 256d
    - Structure Branch: PDB → GVP-GNN (3 layers) → 100d
    - Fusion:           Feature Tokenization → Transformer Classifier → 6 classes
    - Auxiliary Loss:   NT-Xent contrastive loss for cross-modal alignment
"""

__version__ = "1.0.0"
__author__ = "GeoMEL Team"

from .model import GeoMEL
from .dataset import GVPDataset, collate_fn
from .module import (
    GVP,
    GVPConv,
    GVPEncoder,
    AttentionPooling,
    SeqEncoderV6,
    TransformerClassifierV6,
)
from .trainer import Trainer
from .utils import (
    extract_label,
    pdb_to_graph,
    build_graphs,
    FocalLossWithSmoothing,
    NTXentLoss,
    evaluate_model,
    compute_all_metrics,
)

CLASS_NAMES = ["Non-Effector", "T1SE", "T2SE", "T3SE", "T4SE", "T6SE"]

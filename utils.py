"""
Utility functions for GeoMEL.

Includes:
    - Label extraction from FASTA headers
    - PDB to PyG graph conversion
    - Graph building pipeline
    - Focal Loss with Label Smoothing
    - NT-Xent contrastive loss
    - Model evaluation and metrics computation
"""

import os
import re
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio import SeqIO

try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    from Bio.SeqUtils import seq1 as three_to_one

from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Non-Effector", "T1SE", "T2SE", "T3SE", "T4SE", "T6SE"]
NUM_CLASSES = 6

LABEL_PATTERNS = [
    (r"T1SE|T1SS|Type.?1|type.?1", 1),
    (r"T2SE|T2SS|Type.?2|type.?2", 2),
    (r"T3SE|T3SS|Type.?3|type.?3", 3),
    (r"T4SE|T4SS|Type.?4|type.?4", 4),
    (r"T6SE|T6SS|Type.?6|type.?6", 5),
    (r"Non.?[Ee]ffector|non.?effector|negative|NEG", 0),
]

AA_IDX = {a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWYX")}


# ── Label Extraction ─────────────────────────────────────────────────────────


def extract_label(header: str) -> int | None:
    """Extract effector type label from a FASTA description line.

    Args:
        header: FASTA record description string.

    Returns:
        Integer label (0-5) or None if no pattern matches.
    """
    for pattern, idx in LABEL_PATTERNS:
        if re.search(pattern, header, re.IGNORECASE):
            return idx
    return None


# ── PDB → PyG Graph ──────────────────────────────────────────────────────────


def pdb_to_graph(pdb_path: str) -> Data | None:
    """Convert a PDB file to a PyTorch Geometric ``Data`` object.

    Extracts geometric node features (bond lengths, angles, amino acid type)
    and vector features (local coordinate frames) for each residue, then
    constructs edges based on a 10 Å distance cutoff plus sequential
    neighbours.

    Args:
        pdb_path: Path to a ``.pdb`` file.

    Returns:
        A ``torch_geometric.data.Data`` object, or ``None`` on failure.
    """
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("p", pdb_path)
    except Exception:
        return None

    residues = [
        r
        for m in struct
        for c in m
        for r in c
        if is_aa(r, standard=True) and "CA" in r
    ]
    if len(residues) < 5:
        return None

    node_s, node_v, ca_coords = [], [], []
    for res in residues:
        try:
            n = res["N"].get_coord()
            ca = res["CA"].get_coord()
            c = res["C"].get_coord()
            ca_coords.append(ca)

            v1, v2 = n - ca, c - ca
            angle = np.arccos(
                np.clip(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8),
                    -1,
                    1,
                )
            )
            aa_idx = AA_IDX.get(three_to_one(res.get_resname()), 20)
            node_s.append(
                [
                    np.linalg.norm(c - n) / 10,
                    np.linalg.norm(ca - n) / 10,
                    np.linalg.norm(c - ca) / 10,
                    angle / np.pi,
                    np.sin(angle),
                    aa_idx / 20,
                ]
            )
            u1 = (c - ca) / (np.linalg.norm(c - ca) + 1e-8)
            u2 = np.cross(n - ca, c - ca)
            u2 /= np.linalg.norm(u2) + 1e-8
            node_v.append([u1, u2, np.cross(u1, u2)])
        except Exception:
            node_s.append([0] * 6)
            node_v.append([[0, 0, 0]] * 3)
            ca_coords.append([0, 0, 0])

    ca_coords = np.array(ca_coords)
    N = len(ca_coords)
    diff = ca_coords[:, None] - ca_coords[None, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    mask = (dist < 10) & (dist > 0)
    for i in range(N - 1):
        mask[i, i + 1] = mask[i + 1, i] = True
    src, dst = np.where(mask)
    if len(src) == 0:
        return None

    d = dist[src, dst]
    edge_s = np.exp(-((d[:, None] - np.linspace(0, 20, 32)) ** 2) / 2).astype(
        np.float32
    )
    direction = diff[src, dst]
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8

    return Data(
        node_s=torch.tensor(node_s, dtype=torch.float),
        node_v=torch.tensor(node_v, dtype=torch.float),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_s=torch.tensor(edge_s),
        edge_v=torch.tensor(direction[:, None, :].astype(np.float32)),
        num_nodes=N,
    )


def build_graphs(
    pdb_dirs: list[str] | str,
    target_ids: set,
    output_path: str,
    name: str = "graphs",
) -> dict:
    """Build (or load cached) a dictionary mapping protein IDs to PyG graphs.

    Args:
        pdb_dirs: One or more directories containing ``.pdb`` files.
        target_ids: Set of protein IDs to process.
        output_path: Path for caching the result as a ``.pt`` file.
        name: Display name for the progress bar.

    Returns:
        ``dict[str, Data]`` mapping protein ID → graph.
    """
    if os.path.exists(output_path):
        return torch.load(output_path, weights_only=False)

    if isinstance(pdb_dirs, str):
        pdb_dirs = [pdb_dirs]

    pdb_files: dict[str, str] = {}
    for d in pdb_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".pdb"):
                    pid = os.path.splitext(f)[0]
                    if pid not in pdb_files:
                        pdb_files[pid] = os.path.join(d, f)

    graphs: dict[str, Data] = {}
    for pid in tqdm(target_ids, desc=name):
        for tp in [pid, pid.replace("|", "_").replace("~", "_").replace("/", "_")]:
            if tp in pdb_files:
                g = pdb_to_graph(pdb_files[tp])
                if g:
                    graphs[pid] = g
                break

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graphs, output_path)
    return graphs


# ── Loss Functions ───────────────────────────────────────────────────────────


class FocalLossWithSmoothing(nn.Module):
    """Focal Loss with Label Smoothing.

    Combines focal weighting (γ=2 by default) to focus on hard/minority
    examples with label smoothing to improve generalisation.

    Args:
        gamma: Focal exponent.
        smoothing: Label smoothing factor ε.
    """

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(
                logits, self.smoothing / (n_cls - 1)
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(-1)

        probs = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        focal_weight = (1 - probs) ** self.gamma

        return (focal_weight * loss).mean()


class NTXentLoss(nn.Module):
    """Normalised Temperature-scaled Cross-Entropy (NT-Xent) Loss.

    Aligns sequence and structure embeddings for the same protein in a
    shared embedding space using a symmetric contrastive objective.

    Args:
        temperature: Softmax temperature τ.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, z_seq: torch.Tensor, z_struct: torch.Tensor
    ) -> torch.Tensor:
        B = z_seq.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z_seq.device)

        sim = torch.mm(z_seq, z_struct.t()) / self.temperature
        labels = torch.arange(B, device=z_seq.device)

        loss_s2t = F.cross_entropy(sim, labels)
        loss_t2s = F.cross_entropy(sim.t(), labels)
        return (loss_s2t + loss_t2s) / 2


# ── Evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and collect ground-truth labels, predictions, and probs.

    Args:
        model: The GeoMEL model (set to eval mode internally).
        loader: DataLoader yielding collated batches.
        device: Target device.

    Returns:
        ``(y_true, y_pred, y_prob)`` as numpy arrays.
    """
    model.eval()
    preds, labels, probs_list = [], [], []
    for batch in loader:
        if batch is None:
            continue
        for k in batch:
            if k not in ("id", "id_list"):
                batch[k] = batch[k].to(device)
        with torch.cuda.amp.autocast():
            logits, _, _ = model(batch)
        prob = F.softmax(logits.float(), dim=-1)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())
        probs_list.append(prob.cpu().numpy())

    probs_all = (
        np.concatenate(probs_list, axis=0) if probs_list else np.array([])
    )
    return np.array(labels), np.array(preds), probs_all


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
    num_classes: int = NUM_CLASSES,
) -> dict:
    """Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probability matrix (optional, for AUC).
        class_names: List of class display names.
        num_classes: Total number of classes.

    Returns:
        Dictionary of metric name → value.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    m: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, cn in enumerate(class_names):
        if i < len(per_f1):
            m[f"f1_{cn}"] = per_f1[i]

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            m["auc_macro"] = roc_auc_score(
                y_bin, y_prob, average="macro", multi_class="ovr"
            )
            m["auc_weighted"] = roc_auc_score(
                y_bin, y_prob, average="weighted", multi_class="ovr"
            )
            for i, cn in enumerate(class_names):
                if y_bin[:, i].sum() > 0:
                    m[f"auc_{cn}"] = roc_auc_score(y_bin[:, i], y_prob[:, i])
        except Exception:
            pass

    return m

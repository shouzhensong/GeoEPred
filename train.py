import argparse
import os
import pickle
from collections import Counter

import torch
from Bio import SeqIO

from clef_gvp import CLASS_NAMES
from clef_gvp.utils import extract_label, build_graphs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GeoMEL — Geometric Multi-modal Effector Learner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data paths ────────────────────────────────────────────────────────
    g = p.add_argument_group("Data Paths")
    g.add_argument(
        "--train_fasta", type=str, required=True,
        help="Path to training FASTA file.",
    )
    g.add_argument(
        "--test_fasta", type=str, required=True,
        help="Path to test FASTA file.",
    )
    g.add_argument(
        "--train_pdb_dirs", type=str, nargs="+", required=True,
        help="Directories containing training PDB files.",
    )
    g.add_argument(
        "--test_pdb_dirs", type=str, nargs="+", required=True,
        help="Directories containing test PDB files.",
    )
    g.add_argument(
        "--output_dir", type=str, default="./data/processed",
        help="Directory for cached processed data (labels, graphs, ESM features).",
    )
    g.add_argument(
        "--train_esm_file", type=str, default=None,
        help="Path to training ESM features pickle. "
             "Default: <output_dir>/features_esm2_t36_3B/train_esm.pkl",
    )
    g.add_argument(
        "--test_esm_file", type=str, default=None,
        help="Path to test ESM features pickle. "
             "Default: <output_dir>/features_esm2_t36_3B/test_esm.pkl",
    )
    g.add_argument(
        "--results_dir", type=str, default="./results",
        help="Directory for saving training results and checkpoints.",
    )

    # ── Architecture hyper-parameters ─────────────────────────────────────
    g = p.add_argument_group("Architecture")
    g.add_argument("--esm_dim", type=int, default=2560, help="ESM-2 embedding dimension.")
    g.add_argument("--seq_hid", type=int, default=256, help="Sequence branch hidden dimension.")
    g.add_argument("--struct_hid", type=int, default=100, help="Structure branch hidden dimension.")
    g.add_argument("--cls_d_model", type=int, default=128, help="Transformer classifier dimension.")
    g.add_argument("--cls_n_heads", type=int, default=4, help="Number of attention heads.")
    g.add_argument("--cls_n_layers", type=int, default=2, help="Number of Transformer encoder layers.")
    g.add_argument("--num_classes", type=int, default=6, help="Number of output classes.")

    # ── Training hyper-parameters ─────────────────────────────────────────
    g = p.add_argument_group("Training")
    g.add_argument("--gpu_id", type=int, default=0, help="GPU device ID.")
    g.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    g.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    g.add_argument("--num_epochs", type=int, default=80, help="Epochs per fold.")
    g.add_argument("--n_folds", type=int, default=5, help="Number of CV folds.")
    g.add_argument("--contrastive_weight", type=float, default=0.1, help="NT-Xent loss weight λ.")
    g.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing ε.")

    return p.parse_args()


def load_data(args: argparse.Namespace) -> argparse.Namespace:
    """Load and prepare all data, attaching results to ``args``."""

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Resolve ESM paths ─────────────────────────────────────────────────
    if args.train_esm_file is None:
        args.train_esm_file = os.path.join(
            args.output_dir, "features_esm2_t36_3B", "train_esm.pkl"
        )
    if args.test_esm_file is None:
        args.test_esm_file = os.path.join(
            args.output_dir, "features_esm2_t36_3B", "test_esm.pkl"
        )

    # ── Labels ────────────────────────────────────────────────────────────
    train_label_file = os.path.join(args.output_dir, "train_labels.pkl")
    test_label_file = os.path.join(args.output_dir, "test_labels.pkl")

    for fp, fa in [(train_label_file, args.train_fasta),
                   (test_label_file, args.test_fasta)]:
        if not os.path.exists(fp):
            labels = {}
            for r in SeqIO.parse(fa, "fasta"):
                lbl = extract_label(r.description)
                if lbl is not None:
                    labels[r.id] = lbl
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "wb") as f:
                pickle.dump(labels, f)
            print(f"Saved labels: {fp} ({len(labels)} entries)")

    with open(train_label_file, "rb") as f:
        args.train_labels = pickle.load(f)
    with open(test_label_file, "rb") as f:
        args.test_labels = pickle.load(f)

    # ── Graphs ────────────────────────────────────────────────────────────
    train_graph_file = os.path.join(args.output_dir, "train_graphs.pt")
    test_graph_file = os.path.join(args.output_dir, "test_graphs.pt")

    train_ids = set(args.train_labels)
    test_ids = set(args.test_labels)

    args.train_graphs = build_graphs(
        args.train_pdb_dirs, train_ids, train_graph_file, "Train graphs"
    )
    args.test_graphs = build_graphs(
        args.test_pdb_dirs, test_ids, test_graph_file, "Test graphs"
    )

    # ── ESM features ──────────────────────────────────────────────────────
    print(f"Loading ESM features: {args.train_esm_file}")
    with open(args.train_esm_file, "rb") as f:
        args.train_esm = pickle.load(f)
    print(f"Loading ESM features: {args.test_esm_file}")
    with open(args.test_esm_file, "rb") as f:
        args.test_esm = pickle.load(f)

    # ── Valid IDs (intersection of all three sources) ─────────────────────
    args.train_valid = sorted(
        set(args.train_labels) & set(args.train_esm) & set(args.train_graphs)
    )
    args.test_valid = sorted(
        set(args.test_labels) & set(args.test_esm) & set(args.test_graphs)
    )

    for tag, labels in [("Train", args.train_labels), ("Test", args.test_labels)]:
        c = Counter(labels.values())
        dist = ", ".join(f"{CLASS_NAMES[i]}={c.get(i, 0)}" for i in range(args.num_classes))
        print(f"  {tag}: {dist}")

    return args


def main():
    args = parse_args()
    args = load_data(args)

    from clef_gvp.trainer import Trainer

    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__":
    main()

"""
Trainer for GeoMEL.

Provides a ``Trainer`` class that handles:
    - K-Fold stratified cross-validation
    - Weighted random sampling for class imbalance
    - Mixed-precision training with gradient scaling
    - Cosine annealing warm-restart LR scheduling
    - Focal + contrastive loss optimisation
    - Best-model checkpointing per fold (based on validation F1)
      once, after all folds complete, using the best-fold checkpoint
"""

import gc
import json
import os
import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import GVPDataset, collate_fn
from .model import GeoMEL
from .utils import (
    CLASS_NAMES,
    NUM_CLASSES,
    FocalLossWithSmoothing,
    NTXentLoss,
    compute_all_metrics,
    evaluate_model,
)


class Trainer:
    """End-to-end trainer with K-Fold cross-validation.

    Training flow (no data leakage):
        1. For each fold, train on the training split and monitor
           **validation-set F1** to select the best epoch checkpoint.
        2. After **all** folds finish, select the fold with the highest
           best-validation-F1.
        3. Load that fold's best checkpoint and evaluate it **once** on the
           held-out independent test set.

    Args:
        cfg: Namespace or dict-like object with the following attributes:

            **Data paths**

            - ``train_esm``       — dict[str, ndarray]
            - ``test_esm``        — dict[str, ndarray]
            - ``train_labels``    — dict[str, int]
            - ``test_labels``     — dict[str, int]
            - ``train_graphs``    — dict[str, Data]
            - ``test_graphs``     — dict[str, Data]
            - ``train_valid``     — list[str]
            - ``test_valid``      — list[str]

            **Architecture hyper-parameters**

            - ``esm_dim``         — int (default 2560)
            - ``seq_hid``         — int (default 256)
            - ``struct_hid``      — int (default 100)
            - ``cls_d_model``     — int (default 128)
            - ``cls_n_heads``     — int (default 4)
            - ``cls_n_layers``    — int (default 2)
            - ``num_classes``     — int (default 6)

            **Training hyper-parameters**

            - ``batch_size``           — int (default 16)
            - ``learning_rate``        — float (default 1e-4)
            - ``num_epochs``           — int (default 80)
            - ``n_folds``              — int (default 5)
            - ``contrastive_weight``   — float (default 0.1)
            - ``label_smoothing``      — float (default 0.05)
            - ``gpu_id``               — int (default 0)
            - ``results_dir``          — str (default "./results")
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.device = torch.device(
            f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        os.makedirs(cfg.results_dir, exist_ok=True)

    # ── Single fold (NO test-set access) ─────────────────────────────────

    def _train_one_fold(
        self,
        fold: int,
        train_ids_fold: list[str],
        val_ids_fold: list[str],
    ) -> dict:
        cfg = self.cfg
        fold_dir = os.path.join(cfg.results_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # DataLoaders
        fold_train_ds = GVPDataset(
            cfg.train_esm, cfg.train_labels, cfg.train_graphs, train_ids_fold
        )
        fold_val_ds = GVPDataset(
            cfg.train_esm, cfg.train_labels, cfg.train_graphs, val_ids_fold
        )

        fold_targets = [cfg.train_labels[p] for p in train_ids_fold]
        cnt = torch.bincount(
            torch.tensor(fold_targets), minlength=cfg.num_classes
        )
        w = 1.0 / (cnt.float() + 1e-6)
        sampler = WeightedRandomSampler(
            w[torch.tensor(fold_targets)], len(fold_targets), replacement=True
        )

        train_loader = DataLoader(
            fold_train_ds,
            cfg.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            fold_val_ds, cfg.batch_size, collate_fn=collate_fn, num_workers=0
        )

        print(f"  Train: {len(train_ids_fold)}, Val: {len(val_ids_fold)}")

        # Model
        model = GeoMEL(
            cfg.num_classes,
            cfg.esm_dim,
            cfg.seq_hid,
            cfg.struct_hid,
            cfg.cls_d_model,
            cfg.cls_n_heads,
            cfg.cls_n_layers,
        ).to(self.device)

        cls_criterion = FocalLossWithSmoothing(
            gamma=2.0, smoothing=cfg.label_smoothing
        )
        contrast_criterion = NTXentLoss(temperature=0.1)
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4
        )
        scaler = GradScaler()
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2
        )

        history = {"loss": [], "val_f1": [], "val_auc": []}
        best_val_f1 = 0.0
        best_val_metrics: dict = {}
        best_val_y_true, best_val_y_pred = None, None
        best_val_probs = None

        for epoch in range(cfg.num_epochs):
            # ── Train ──
            model.train()
            loss_sum, nb = 0.0, 0
            for batch in train_loader:
                if batch is None:
                    continue
                for k in batch:
                    if k not in ("id", "id_list"):
                        batch[k] = batch[k].to(self.device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logits, z_seq, z_struct = model(batch)
                    cls_loss = cls_criterion(logits, batch["label"])
                    con_loss = contrast_criterion(z_seq, z_struct)
                    loss = cls_loss + cfg.contrastive_weight * con_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item()
                nb += 1
            scheduler.step()

            # ── Validate (only validation set, NO test set) ──
            val_true, val_pred, val_prob = evaluate_model(
                model, val_loader, self.device
            )
            val_f1 = f1_score(
                val_true, val_pred, average="macro", zero_division=0
            )
            try:
                val_auc = roc_auc_score(
                    label_binarize(
                        val_true, classes=list(range(cfg.num_classes))
                    ),
                    val_prob,
                    average="macro",
                    multi_class="ovr",
                )
            except Exception:
                val_auc = 0.0

            history["loss"].append(loss_sum / max(nb, 1))
            history["val_f1"].append(val_f1)
            history["val_auc"].append(val_auc)

            # Checkpoint on best validation F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_metrics = compute_all_metrics(
                    val_true, val_pred, val_prob
                )
                best_val_y_true = val_true.copy()
                best_val_y_pred = val_pred.copy()
                best_val_probs = val_prob.copy()
                torch.save(
                    model.state_dict(),
                    os.path.join(fold_dir, "best_model.pt"),
                )

            if (epoch + 1) % 10 == 0:
                star = " ★" if val_f1 >= best_val_f1 else ""
                print(
                    f"    Epoch {epoch+1:02d} | Loss: {loss_sum/max(nb,1):.4f} | "
                    f"Val-F1: {val_f1:.4f} (AUC: {val_auc:.4f}){star}"
                )

        # ── Save fold artefacts (validation only) ──
        val_cm = confusion_matrix(best_val_y_true, best_val_y_pred)
        np.save(os.path.join(fold_dir, "val_confusion_matrix.npy"), val_cm)

        val_report = classification_report(
            best_val_y_true,
            best_val_y_pred,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
        with open(
            os.path.join(fold_dir, "val_classification_report.txt"), "w"
        ) as f:
            f.write(f"Fold {fold} — Validation Set\n{'=' * 60}\n{val_report}")

        fold_result = {
            "fold": fold,
            "train_size": len(train_ids_fold),
            "val_size": len(val_ids_fold),
            "best_val_f1": best_val_f1,
            "best_val_metrics": best_val_metrics,
            "history": history,
        }
        with open(os.path.join(fold_dir, "fold_result.json"), "w") as f:
            json.dump(fold_result, f, indent=2)

        np.savez(
            os.path.join(fold_dir, "roc_data.npz"),
            val_y_true=best_val_y_true,
            val_probs=best_val_probs,
        )

        print(
            f"  ✓ Fold {fold} done | Best Val-F1: {best_val_f1:.4f} | "
            f"Val-AUC: {best_val_metrics.get('auc_macro', 0):.4f} | "
            f"Val-Acc: {best_val_metrics.get('accuracy', 0):.4f} | "
            f"Val-MCC: {best_val_metrics.get('mcc', 0):.4f}"
        )

        del model, optimizer, scaler
        torch.cuda.empty_cache()
        gc.collect()

        return fold_result

    # ── Test evaluation (called ONCE after all folds) ────────────────────

    def _evaluate_test_set(self, best_fold: int) -> dict:
        """Load the best-fold checkpoint and evaluate on the independent test set.

        This is the **only** place in the entire pipeline where the test set
        is accessed.

        Args:
            best_fold: The fold number whose checkpoint to use.

        Returns:
            Dictionary with test metrics, predictions, and probabilities.
        """
        cfg = self.cfg
        fold_dir = os.path.join(cfg.results_dir, f"fold_{best_fold}")
        ckpt_path = os.path.join(fold_dir, "best_model.pt")

        print(f"\n{'=' * 60}")
        print(f"  Evaluating independent test set with Fold {best_fold} checkpoint")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'=' * 60}")

        # Build test loader
        test_ds = GVPDataset(
            cfg.test_esm, cfg.test_labels, cfg.test_graphs, cfg.test_valid
        )
        test_loader = DataLoader(
            test_ds, cfg.batch_size, collate_fn=collate_fn, num_workers=0
        )

        # Load model
        model = GeoMEL(
            cfg.num_classes,
            cfg.esm_dim,
            cfg.seq_hid,
            cfg.struct_hid,
            cfg.cls_d_model,
            cfg.cls_n_heads,
            cfg.cls_n_layers,
        ).to(self.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

        # Evaluate
        test_true, test_pred, test_prob = evaluate_model(
            model, test_loader, self.device
        )
        test_metrics = compute_all_metrics(test_true, test_pred, test_prob)

        # Save test artefacts
        cm = confusion_matrix(test_true, test_pred)
        np.save(os.path.join(cfg.results_dir, "test_confusion_matrix.npy"), cm)

        test_report = classification_report(
            test_true,
            test_pred,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
        report_path = os.path.join(cfg.results_dir, "test_classification_report.txt")
        with open(report_path, "w") as f:
            f.write(
                f"GeoMEL — Independent Test Set Evaluation\n"
                f"Best fold: {best_fold}\n"
                f"{'=' * 60}\n{test_report}"
            )

        np.savez(
            os.path.join(cfg.results_dir, "test_roc_data.npz"),
            test_y_true=test_true,
            test_probs=test_prob,
        )

        # Print test results
        print(f"\n  Test Results (Fold {best_fold} checkpoint):")
        print(f"    Accuracy:    {test_metrics['accuracy']:.4f}")
        print(f"    F1-Macro:    {test_metrics['f1_macro']:.4f}")
        print(f"    F1-Weighted: {test_metrics['f1_weighted']:.4f}")
        print(f"    Precision:   {test_metrics['precision_macro']:.4f}")
        print(f"    Recall:      {test_metrics['recall_macro']:.4f}")
        print(f"    MCC:         {test_metrics['mcc']:.4f}")
        print(f"    AUC-Macro:   {test_metrics.get('auc_macro', 0):.4f}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "best_fold": best_fold,
            "test_metrics": test_metrics,
            "test_y_true": test_true.tolist(),
            "test_y_pred": test_pred.tolist(),
        }

    # ── K-Fold entry point ───────────────────────────────────────────────

    def run(self) -> tuple[list[dict], dict]:
        """Execute K-Fold cross-validation training and final test evaluation.

        Returns:
            ``(fold_results, test_result)``
                - ``fold_results``: List of per-fold result dictionaries
                  (validation metrics only).
                - ``test_result``: Test-set evaluation result from the best fold.
        """
        cfg = self.cfg

        train_ids_arr = np.array(cfg.train_valid)
        train_labels_arr = np.array(
            [cfg.train_labels[p] for p in cfg.train_valid]
        )

        skf = StratifiedKFold(
            n_splits=cfg.n_folds, shuffle=True, random_state=42
        )

        print("#" * 60)
        print(f"# GeoMEL — {cfg.n_folds}-Fold Cross Validation")
        print(f"# {cfg.num_epochs} epochs per fold")
        print("#" * 60)

        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_ids_arr, train_labels_arr), 1
        ):
            print(f"\n{'▓' * 60}")
            print(f"▓  Fold {fold}/{cfg.n_folds}")
            print(f"{'▓' * 60}")

            fold_train_ids = train_ids_arr[train_idx].tolist()
            fold_val_ids = train_ids_arr[val_idx].tolist()

            val_dist = Counter(
                [cfg.train_labels[p] for p in fold_val_ids]
            )
            print(
                f"  Val distribution: "
                + ", ".join(
                    f"{CLASS_NAMES[i]}={val_dist.get(i, 0)}"
                    for i in range(cfg.num_classes)
                )
            )

            result = self._train_one_fold(fold, fold_train_ids, fold_val_ids)
            fold_results.append(result)

        # ── Select best fold by validation F1 ────────────────────────────
        best_fold_idx = int(
            np.argmax([r["best_val_f1"] for r in fold_results])
        )
        best_fold = fold_results[best_fold_idx]["fold"]

        print(f"\n{'=' * 60}")
        print(f"All {cfg.n_folds} folds complete.")
        print(f"Best fold by Val-F1: Fold {best_fold} "
              f"(F1={fold_results[best_fold_idx]['best_val_f1']:.4f})")
        print(f"{'=' * 60}")

        # ── Evaluate test set ONCE with best-fold checkpoint ─────────────
        test_result = self._evaluate_test_set(best_fold)

        # ── Summary ──────────────────────────────────────────────────────
        self._print_summary(fold_results, test_result)
        self._save_final_results(fold_results, test_result)

        return fold_results, test_result

    # ── Helpers ───────────────────────────────────────────────────────────

    def _print_summary(
        self, fold_results: list[dict], test_result: dict
    ) -> None:
        cfg = self.cfg
        metric_keys = [
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
            "mcc",
            "auc_macro",
        ]
        metric_names = [
            "Accuracy",
            "F1-Macro",
            "F1-Weighted",
            "Precision",
            "Recall",
            "MCC",
            "AUC-Macro",
        ]

        # ── Validation CV summary ──
        print("\n" + "=" * 70)
        print(
            f"GeoMEL — {cfg.n_folds}-Fold CV Results (Validation Set)"
        )
        print("=" * 70)

        header = f"{'Fold':<6}" + "".join(f"{n:<12}" for n in metric_names)
        print(header)
        print("-" * 70)

        fold_vals = {k: [] for k in metric_keys}
        for res in fold_results:
            vm = res["best_val_metrics"]
            line = f"  {res['fold']:<4}"
            for k in metric_keys:
                v = vm.get(k, 0)
                fold_vals[k].append(v)
                line += f"{v:<12.4f}"
            print(line)

        print("-" * 70)
        mean_line = f"  {'Mean':<4}"
        std_line = f"  {'Std':<4}"
        for k in metric_keys:
            mean_line += f"{np.mean(fold_vals[k]):<12.4f}"
            std_line += f"{np.std(fold_vals[k]):<12.4f}"
        print(mean_line)
        print(std_line)

        # ── Test result ──
        tm = test_result["test_metrics"]
        print(f"\n{'=' * 70}")
        print(
            f"GeoMEL — Independent Test Set "
            f"(Fold {test_result['best_fold']} checkpoint)"
        )
        print("=" * 70)
        for k, n in zip(metric_keys, metric_names):
            print(f"  {n:<18} {tm.get(k, 0):.4f}")

    def _save_final_results(
        self, fold_results: list[dict], test_result: dict
    ) -> None:
        cfg = self.cfg
        metric_keys = [
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
            "mcc",
            "auc_macro",
        ]

        fold_vals = {k: [] for k in metric_keys}
        for res in fold_results:
            for k in metric_keys:
                fold_vals[k].append(res["best_val_metrics"].get(k, 0))

        final = {
            "model": "GeoMEL",
            "cross_validation": f"{cfg.n_folds}-Fold Stratified",
            "architecture": {
                "seq_encoder": "ESM-2 (36L, 3B) → SeqEncoder (AttentionPooling)",
                "struct_encoder": "GVP-GNN (3 layers)",
                "fusion": "Feature Tokenization + Concatenation",
                "classifier": (
                    f"Transformer ({cfg.cls_d_model}d, "
                    f"{cfg.cls_n_heads}heads, {cfg.cls_n_layers}layers)"
                ),
                "aux_loss": f"NT-Xent (λ={cfg.contrastive_weight})",
            },
            "hyperparameters": {
                "esm_dim": cfg.esm_dim,
                "seq_hid": cfg.seq_hid,
                "struct_hid": cfg.struct_hid,
                "cls_d_model": cfg.cls_d_model,
                "cls_n_heads": cfg.cls_n_heads,
                "batch_size": cfg.batch_size,
                "lr": cfg.learning_rate,
                "epochs": cfg.num_epochs,
                "label_smoothing": cfg.label_smoothing,
                "n_folds": cfg.n_folds,
            },
            "cv_val_mean_metrics": {
                k: float(np.mean(fold_vals[k])) for k in metric_keys
            },
            "cv_val_std_metrics": {
                k: float(np.std(fold_vals[k])) for k in metric_keys
            },
            "per_fold_val_metrics": [
                r["best_val_metrics"] for r in fold_results
            ],
            "best_fold": test_result["best_fold"],
            "test_metrics": test_result["test_metrics"],
        }

        path = os.path.join(cfg.results_dir, "geomel_results.json")
        with open(path, "w") as f:
            json.dump(final, f, indent=2)
        print(f"\nResults saved to: {path}")

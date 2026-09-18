"""
train_fedprox_fednova.py — Real federated training run reproducing the
top-performing published approach found for federated melanoma
classification: EfficientNet-B0 backbone + hybrid FedProx (client) /
FedNova (server) optimization.

Paper reference: Springer ICACECS 2025, "Federated Learning for Skin
Cancer Detection with Optimized Client Selection and Hybrid Aggregation
Strategies" — reports ~93.56% accuracy on ISIC-derived data with this
combination.

This is a REAL federated training loop (real gradients, real per-hospital
image data, real weight aggregation across rounds) — not a metadata-only
simulation like server/simulation/simulation.py, which only models client
selection and straggler timing without any actual model training.

Data source
-----------
Uses this repo's existing 10-hospital Non-IID partition layout, unchanged:

    <data_root>/<HospitalName>/metadata_train.csv
    <data_root>/<HospitalName>/metadata_test.csv
    <data_root>/<HospitalName>/images/Train/<isic_id>.jpg
    <data_root>/<HospitalName>/images/Test/<isic_id>.jpg

Reuses, unmodified:
    client/dataset.py      -> MelanomaDataset (image + 13-d metadata + label)
    client/focal_loss.py   -> ClassWeightedFocalLoss, compute_class_weights

Loaded via a controlled sys.path insertion (this experiment's own directory
first, so `import model` resolves to this folder's EfficientNet-B0 variant,
not client/model.py's canonical EfficientNet-B3) rather than importlib's
spec_from_file_location loader — the latter creates modules under a
synthetic, non-re-importable name, which breaks Windows' spawn-based
DataLoader multiprocessing (num_workers > 0) when it tries to pickle
dataset objects across the process boundary.

Simplifications versus the exact paper setup (documented, not hidden)
-----------------------------------------------------------------------
1. Client selection: this run uses FULL PARTICIPATION (all 10 hospitals
   every round). The source paper describes a "class-aware, class-diverse"
   client selection policy but does not publish its exact scoring formula,
   so it cannot be replicated bit-for-bit. This repo already has a
   published, fully-specified alternative (server/selector.py's Boltzmann
   exposure-aware selector, docs/FedMelanoma_Blueprint_v4.md Section 3.8-3.9)
   which would be the natural next integration step.
2. FedProx and FedNova themselves ARE precisely, mathematically specified
   in their original papers (Li et al. 2020; Wang et al. 2020) and are
   implemented here exactly per those specifications — see
   fedprox_fednova.py's docstrings for the formulas.

Usage
-----
    python train_fedprox_fednova.py --data-root "C:/path/to/hospital_partitions"
    python train_fedprox_fednova.py --data-root ./hospital_partitions --rounds 3 --local-epochs 1 --limit-per-hospital 20   # fast smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    balanced_accuracy_score,
)

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
# NOTE ON WINDOWS MULTIPROCESSING (num_workers > 0):
#   torch's DataLoader workers are spawned as fresh child processes on
#   Windows (no fork()), which re-import this script's module graph to
#   reconstruct picklable objects. That means dataset.py / focal_loss.py
#   MUST be reachable via ordinary sys.path imports (real, re-importable
#   module names) rather than importlib.util.spec_from_file_location()
#   loaded under a synthetic name — pickle needs to `import <module_name>`
#   by name in the child process, and a synthetic name isn't importable
#   there even though it exists in the parent's sys.modules. Using
#   sys.path.insert() here (top-level code, so it re-runs when the child
#   re-executes this script) is the standard fix for this exact failure
#   mode (PicklingError: "import of module '<name>' failed").
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
CLIENT_DIR = REPO_ROOT / "client"

# THIS_DIR must win over CLIENT_DIR for `import model` (this experiment's
# EfficientNet-B0 model.py must not be shadowed by client/model.py's
# EfficientNet-B3 canonical version) — insert CLIENT_DIR first, THIS_DIR
# last, so THIS_DIR ends up first in sys.path.
sys.path.insert(0, str(CLIENT_DIR))
sys.path.insert(0, str(THIS_DIR))

from model import FedMelanomaModelB0  # noqa: E402
from fedprox_fednova import (  # noqa: E402
    split_state_keys,
    local_train_fedprox,
    fednova_aggregate,
    clone_state_dict_cpu,
)
from dataset import MelanomaDataset  # noqa: E402
from focal_loss import ClassWeightedFocalLoss, compute_class_weights  # noqa: E402


# ---------------------------------------------------------------------------
# The 10 canonical hospital partitions (matches without_federated/client/train.py)
# ---------------------------------------------------------------------------
HOSPITAL_NAMES: List[str] = [
    "CityGeneral_Multispecialty_Hospital",
    "GoldenSun_Public_Health_Screening_Camp",
    "Greenfield_Rural_Community_Health_Center",
    "Lakeside_Regional_Mixed_Hospital",
    "MetroSkin_Dermatology_Research_Clinic",
    "NorthCare_Tertiary_Melanoma_Institute",
    "Pinewood_Oncology_Biopsy_Center",
    "SunRay_Community_Outreach_Clinic",
    "TeleDerm_National_Screening_Network",
    "Westbridge_Academic_Dermoscopy_Unit",
]

POSITIVE_CLASS_IDX = 1  # Malignant = 1


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, meta, labels in loader:
            images = images.to(device)
            meta = meta.to(device)
            labels = labels.to(device)

            logits = model(images, meta)
            probs = F.softmax(logits, dim=1)[:, POSITIVE_CLASS_IDX]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    accuracy = accuracy_score(y_true, y_pred)
    # Balanced accuracy = (sensitivity + specificity) / 2 = mean of per-class
    # recall. Reported alongside plain accuracy because this dataset is
    # ~17:1 imbalanced (5.55% malignant): an all-benign classifier already
    # scores ~94.4% plain accuracy on this test set, so plain accuracy alone
    # can look strong while barely beating a trivial baseline. Balanced
    # accuracy does not reward that shortcut, since it weights the rare
    # malignant class equally with the common benign class.
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS_IDX, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS_IDX, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS_IDX, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_eval": int(len(y_true)),
    }


# ===========================================================================
# Data loading
# ===========================================================================

def build_hospital_loaders(data_root: Path, batch_size: int, num_workers: int, limit_per_hospital: int | None):
    """
    Build one (train_loader, class_weights, test_dataset) tuple per hospital,
    using the existing hospital_partitions/ layout on disk.
    """
    per_hospital = {}
    test_datasets = []

    for hospital in HOSPITAL_NAMES:
        base = data_root / hospital

        train_csv = base / "metadata_train.csv"
        test_csv = base / "metadata_test.csv"
        train_img_dir = base / "images" / "Train"
        test_img_dir = base / "images" / "Test"

        if not train_csv.exists() or not test_csv.exists():
            print(f"  [WARN] Skipping {hospital}: metadata CSVs not found under {base}")
            continue

        train_ds = MelanomaDataset(str(train_csv), str(train_img_dir))
        test_ds = MelanomaDataset(str(test_csv), str(test_img_dir))

        if limit_per_hospital is not None:
            n = min(limit_per_hospital, len(train_ds))
            train_ds = Subset(train_ds, list(range(n)))
            n_test = min(max(limit_per_hospital // 4, 5), len(test_ds))
            test_ds = Subset(test_ds, list(range(n_test)))

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False,
        )

        # Class weights computed locally from this hospital's own train labels
        # (never transmitted raw — consistent with the rest of this repo).
        raw_df = train_ds.dataset.df if isinstance(train_ds, Subset) else train_ds.df
        indices = train_ds.indices if isinstance(train_ds, Subset) else range(len(raw_df))
        targets = [
            1 if str(raw_df.iloc[i]["diagnosis_1"]).lower() == "malignant" else 0
            for i in indices
        ]
        class_weights = compute_class_weights(torch.tensor(targets))

        per_hospital[hospital] = {
            "train_loader": train_loader,
            "class_weights": class_weights,
            "n_train": len(train_ds),
        }
        test_datasets.append(test_ds)

        mal_rate = 100.0 * (sum(targets) / max(len(targets), 1))
        print(f"  [Data] {hospital}: n_train={len(train_ds):>5}  n_test={len(test_ds):>5}  malignant%={mal_rate:5.2f}")

    global_test_dataset = ConcatDataset(test_datasets)
    return per_hospital, global_test_dataset


# ===========================================================================
# Main federated training loop
# ===========================================================================

def main() -> None:
    # Force line-buffered stdout even when redirected to a file, so progress
    # is visible in real time instead of sitting in Python's output buffer
    # until it fills or the process exits (the cause of an apparently silent
    # hour-long run that was, in fact, progressing normally).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=str, default="hospital_partitions",
                        help="Path to the hospital_partitions directory (10 hospital subfolders).")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds T.")
    parser.add_argument("--local-epochs", type=int, default=2, help="Local epochs E per client per round.")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal term coefficient.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-path", type=str, default=str(THIS_DIR / "best_effnet_b0_fedprox_fednova.pth"))
    parser.add_argument("--log-path", type=str, default=str(THIS_DIR / "training_log.json"))
    parser.add_argument("--limit-per-hospital", type=int, default=None,
                        help="Cap samples per hospital (smoke testing only; omit for a real run).")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a checkpoint .pth saved by this script. Loads its model_state_dict "
                             "as the starting global model instead of a fresh pretrained init, and resumes "
                             "round numbering after its saved round. Use with --resume-log to also carry "
                             "forward that checkpoint's round-by-round history into the new log file.")
    parser.add_argument("--resume-log", type=str, default=None,
                        help="Path to a training_log.json from an earlier interrupted run. Its 'history' "
                             "entries are prepended to this run's history so the final log/report covers "
                             "all rounds, not just the ones completed after resuming.")
    args = parser.parse_args()

    device = torch.device(args.device)
    data_root = Path(args.data_root)

    print("=" * 70)
    print("  EfficientNet-B0 + Hybrid FedProx/FedNova — Federated Training")
    print("  Reproducing: Springer ICACECS 2025 (~93.56% reported accuracy)")
    print("=" * 70)
    print(f"  Data root      : {data_root}")
    print(f"  Device         : {device}")
    print(f"  Rounds (T)     : {args.rounds}")
    print(f"  Local epochs(E): {args.local_epochs}")
    print(f"  FedProx mu     : {args.mu}")
    print(f"  Batch size     : {args.batch_size}")
    print("=" * 70)

    # --- Load per-hospital data -------------------------------------------
    print("\n[Data] Loading 10 hospital partitions...")
    per_hospital, global_test_dataset = build_hospital_loaders(
        data_root, args.batch_size, args.num_workers, args.limit_per_hospital
    )
    if not per_hospital:
        raise RuntimeError(f"No hospital partitions found under {data_root}. Check --data-root.")

    test_loader = DataLoader(
        global_test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )
    print(f"[Data] Global test set: {len(global_test_dataset)} samples across {len(per_hospital)} hospitals.")

    # --- Build global model + local working model -------------------------
    global_model = FedMelanomaModelB0().to(device)
    local_model = FedMelanomaModelB0().to(device)   # reused every client/round; loaded fresh each call
    eval_model = global_model  # evaluation reuses the global model's weights each round

    global_state = clone_state_dict_cpu(global_model.state_dict())
    param_keys, buffer_keys = split_state_keys(global_model)
    print(f"\n[Model] EfficientNet-B0 fusion model: {sum(p.numel() for p in global_model.parameters()):,} total params "
          f"({len(param_keys)} trainable tensors, {len(buffer_keys)} buffer tensors).")

    criterion = ClassWeightedFocalLoss(gamma_f=2.0, reduction="mean")

    best_roc_auc = 0.0
    best_metrics: Dict = {}
    history: List[Dict] = []
    start_round = 1

    # --- Resume from an interrupted run's checkpoint, if requested ---------
    if args.resume_from:
        print(f"\n[Resume] Loading checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        global_state = clone_state_dict_cpu(ckpt["model_state_dict"])
        start_round = int(ckpt["round"]) + 1
        best_metrics = ckpt.get("metrics", {})
        best_roc_auc = float(best_metrics.get("roc_auc", 0.0))
        print(f"[Resume] Checkpoint was round {ckpt['round']} "
              f"(accuracy={best_metrics.get('accuracy', 0)*100:.2f}%, roc_auc={best_roc_auc:.4f}). "
              f"Resuming at round {start_round}.")

    if args.resume_log:
        print(f"[Resume] Loading prior history: {args.resume_log}")
        with open(args.resume_log, "r") as f:
            prior = json.load(f)
        history = list(prior.get("history", []))
        print(f"[Resume] Carried forward {len(history)} completed round(s) of history.")

    if start_round > args.rounds:
        print(f"\n[Resume] Checkpoint round {start_round - 1} already reaches or exceeds "
              f"--rounds {args.rounds}. Nothing to do.")
        return

    total_start = time.time()

    for round_idx in range(start_round, args.rounds + 1):
        round_start = time.time()
        print(f"\n{'-'*60}")
        print(f"  ROUND {round_idx}/{args.rounds}")
        print(f"{'-'*60}")

        client_results = []
        client_names = []

        for hospital, hdata in per_hospital.items():
            result = local_train_fedprox(
                model=local_model,
                global_state_dict=global_state,
                dataloader=hdata["train_loader"],
                criterion=criterion,
                class_weights=hdata["class_weights"].to(device),
                device=device,
                mu=args.mu,
                local_epochs=args.local_epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                param_keys=param_keys,
                buffer_keys=buffer_keys,
                progress_label=f"R{round_idx} {hospital}",
                progress_every=5,
            )
            client_results.append(result)
            client_names.append(hospital)
            print(f"    {hospital:<45} avg_loss={result.avg_loss:.4f}  tau={result.tau:>4}  n={result.n_samples}")

        # --- FedNova aggregation -------------------------------------------
        global_state, tau_eff = fednova_aggregate(global_state, client_results, param_keys, buffer_keys)
        print(f"  [Aggregation] FedNova tau_eff={tau_eff:.2f} "
              f"(effective normalized step count for this round)")

        # --- Evaluate global model ------------------------------------------
        eval_model.load_state_dict(global_state)
        eval_model.to(device)
        metrics = evaluate(eval_model, test_loader, device)
        round_time = time.time() - round_start

        print(
            f"  [Eval] Accuracy={metrics['accuracy']*100:.2f}%  "
            f"BalancedAcc={metrics['balanced_accuracy']*100:.2f}%  "
            f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
            f"(round took {round_time:.1f}s)"
        )

        history.append({
            "round": round_idx,
            "tau_eff": tau_eff,
            "round_time_sec": round_time,
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            "client_avg_losses": {n: r.avg_loss for n, r in zip(client_names, client_results)},
        })

        if metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = metrics["roc_auc"]
            best_metrics = metrics
            torch.save({
                "round": round_idx,
                "model_state_dict": global_state,
                "metrics": metrics,
                "args": vars(args),
            }, args.checkpoint_path)
            print(f"  [Checkpoint] New best ROC-AUC={best_roc_auc:.4f} -> saved to {args.checkpoint_path}")

        # Persist running log after every round (safe to interrupt)
        with open(args.log_path, "w") as f:
            json.dump({"history": history, "best_metrics": best_metrics}, f, indent=2)

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time        : {total_time/60:.1f} min")
    print(f"  Best Accuracy      : {best_metrics.get('accuracy', 0)*100:.2f}%")
    print(f"  Best Precision     : {best_metrics.get('precision', 0):.4f}")
    print(f"  Best Recall        : {best_metrics.get('recall', 0):.4f}")
    print(f"  Best F1            : {best_metrics.get('f1', 0):.4f}")
    print(f"  Best ROC-AUC       : {best_metrics.get('roc_auc', 0):.4f}")
    print(f"  Checkpoint         : {args.checkpoint_path}")
    print(f"  Log                : {args.log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

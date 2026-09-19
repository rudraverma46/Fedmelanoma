"""
train_melanoma.py
=================
Production-quality centralized training script for melanoma binary
classification using FedMelanomaModel, MelanomaDataset, and
ClassWeightedFocalLoss.

Key features
------------
* Comprehensive per-epoch metrics: Accuracy, Precision, Recall, F1,
  ROC-AUC, Confusion Matrix, Classification Report
* Malignant-class (class 1) metrics clearly reported
* Best-model checkpoint saved based on ROC-AUC
* Early stopping with configurable patience (default 5 epochs)
* Fixed train-loss accumulation (average loss per epoch)
* Dataset statistics printed before training begins
* Optional matplotlib plots: train loss / accuracy / ROC-AUC vs epoch
"""
import time
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

# ---------------------------------------------------------------------------
# Project-local imports (unchanged API)
# ---------------------------------------------------------------------------
from dataset import MelanomaDataset
from focal_loss import ClassWeightedFocalLoss, compute_class_weights
from model import FedMelanomaModel

# ---------------------------------------------------------------------------
# sklearn metrics
# ---------------------------------------------------------------------------
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------------------------------------------------------
# Dataset location
# ---------------------------------------------------------------------------
# CHANGED: this used to be the hardcoded relative string "hospital_partitions",
# which only worked if you launched `python train.py` from the exact parent
# directory of that folder. It's now an absolute path constant, so the script
# works no matter which directory you run it from.
#
# Forward slashes are used deliberately (instead of backslashes) because
# Python/Windows both accept "/" in paths, and it avoids any accidental
# backslash-escape issues (e.g. "\t", "\U...") that can occur in a plain
# (non-raw) Windows path string.
DATASET_ROOT: str = "D:/MAIN BACKUP/backup 3/Fedmelanoma/without_federated/client/hospital_partitions"

# ---------------------------------------------------------------------------
# Training hyper-parameters (edit here rather than deep inside main)
# ---------------------------------------------------------------------------
NUM_EPOCHS: int = 100
BATCH_SIZE: int = 8
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-4
FOCAL_GAMMA: float = 2.0
EARLY_STOPPING_PATIENCE: int = 5
MIN_EPOCHS_BEFORE_STOPPING: int = 50
CHECKPOINT_PATH: str = "best_fedmelanoma_model.pth"
POSITIVE_CLASS_IDX: int = 1  # Malignant = 1


# ===========================================================================
# Helper: dataset statistics
# ===========================================================================

def print_dataset_statistics(
    train_targets: List[int],
    test_targets: List[int],
    class_weights: torch.Tensor,
) -> None:
    """Print counts, class distribution, and computed class weights."""
    n_train = len(train_targets)
    n_test = len(test_targets)

    train_arr = np.array(train_targets)
    test_arr  = np.array(test_targets)

    train_benign    = int((train_arr == 0).sum())
    train_malignant = int((train_arr == 1).sum())
    test_benign     = int((test_arr == 0).sum())
    test_malignant  = int((test_arr == 1).sum())

    sep = "=" * 60
    print(sep)
    print("DATASET STATISTICS")
    print(sep)
    print(f"  Train samples  : {n_train:>8,}")
    print(f"    Benign  (0)  : {train_benign:>8,}  "
          f"({100*train_benign/n_train:.1f}%)")
    print(f"    Malignant (1): {train_malignant:>8,}  "
          f"({100*train_malignant/n_train:.1f}%)")
    print(f"  Test  samples  : {n_test:>8,}")
    print(f"    Benign  (0)  : {test_benign:>8,}  "
          f"({100*test_benign/n_test:.1f}%)")
    print(f"    Malignant (1): {test_malignant:>8,}  "
          f"({100*test_malignant/n_test:.1f}%)")
    print(f"  Class weights  : Benign={class_weights[0]:.4f}  "
          f"Malignant={class_weights[1]:.4f}")
    print(sep)
    print()


# ===========================================================================
# Helper: one training epoch
# ===========================================================================

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: ClassWeightedFocalLoss,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    epoch_start: float,
) -> float:
    """Run one full training epoch and return average batch loss."""
    model.train()
    running_loss: float = 0.0
    num_batches: int = 0

    for batch_idx, (images, meta, labels) in enumerate(loader):

        

        if batch_idx % 100 == 0:

            progress = (batch_idx + 1) / len(loader)

            elapsed = time.time() - epoch_start

            eta_epoch = elapsed / progress - elapsed

            print(
                f"Epoch Progress: "
                f"{100*progress:.1f}% | "
                f"ETA Epoch: {eta_epoch/60:.2f} min"
            )
        images = images.to(DEVICE)
        meta = meta.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        

        logits = model(images, meta)



        loss = criterion(
            logits,
            labels,
            class_weights=class_weights,
        )


        loss.backward()



        optimizer.step()

        running_loss += loss.item()
        num_batches += 1


    avg_loss: float = running_loss / max(num_batches, 1)
    return avg_loss


# ===========================================================================
# Helper: validation / evaluation
# ===========================================================================

def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on *loader* and compute all metrics.

    Returns
    -------
    metrics : dict with keys accuracy, precision, recall, f1, roc_auc
    all_labels : ground-truth label array
    all_preds  : predicted label array
    all_probs  : softmax probability for Malignant class (class 1)
    """
    model.eval()

    all_labels: List[int]  = []
    all_preds:  List[int]  = []
    all_probs:  List[float] = []

    with torch.no_grad():
        for images, meta, labels in loader:
            images = images.to(DEVICE)
            meta   = meta.to(DEVICE)
            labels = labels.to(DEVICE)

            logits: torch.Tensor = model(images, meta)

            # Softmax probabilities; take column for Malignant (class 1)
            probs: torch.Tensor = F.softmax(logits, dim=1)[:, POSITIVE_CLASS_IDX]
            preds: torch.Tensor = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_prob  = np.array(all_probs)

    # Core scalar metrics (macro average for multi-class safety; binary here)
    accuracy:  float = accuracy_score(y_true, y_pred)
    precision: float = precision_score(
        y_true, y_pred, pos_label=POSITIVE_CLASS_IDX, zero_division=0
    )
    recall: float = recall_score(
        y_true, y_pred, pos_label=POSITIVE_CLASS_IDX, zero_division=0
    )
    f1: float = f1_score(
        y_true, y_pred, pos_label=POSITIVE_CLASS_IDX, zero_division=0
    )

    # ROC-AUC requires probability scores
    try:
        roc_auc: float = roc_auc_score(y_true, y_prob)
    except ValueError:
        # Edge case: only one class present in labels
        roc_auc = 0.0

    metrics: Dict[str, float] = {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "roc_auc":   roc_auc,
    }

    return metrics, y_true, y_pred, y_prob


# ===========================================================================
# Helper: pretty-print per-epoch metrics
# ===========================================================================

def print_epoch_metrics(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    metrics: Dict[str, float],
) -> None:
    """Print a clean, aligned metrics block for one epoch."""
    sep = "-" * 60
    print(sep)
    print(f"Epoch {epoch}/{total_epochs}")
    print(f"  Train Loss           : {train_loss:.4f}")
    print(f"  Validation Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Malignant Precision  : {metrics['precision']:.4f}")
    print(f"  Malignant Recall     : {metrics['recall']:.4f}")
    print(f"  Malignant F1         : {metrics['f1']:.4f}")
    print(f"  Validation ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(sep)


# ===========================================================================
# Helper: confusion matrix + classification report
# ===========================================================================

def print_detailed_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epoch: int,
) -> None:
    """Print confusion matrix and sklearn classification report."""
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix (Epoch {epoch}):")
    print(f"             Pred Benign  Pred Malignant")
    print(f"  True Benign    {cm[0,0]:>6}          {cm[0,1]:>6}")
    print(f"  True Malignant {cm[1,0]:>6}          {cm[1,1]:>6}")

    print("\n  Classification Report:")
    print(
        classification_report(
            y_true, y_pred,
            target_names=["Benign", "Malignant"],
            zero_division=0,
        )
    )


# ===========================================================================
# Helper: save checkpoint
# ===========================================================================

def save_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: Dict[str, float],
) -> None:
    """Persist model + optimiser state alongside epoch metrics."""
    checkpoint = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy":             metrics["accuracy"],
        "precision":            metrics["precision"],
        "recall":               metrics["recall"],
        "f1":                   metrics["f1"],
        "roc_auc":              metrics["roc_auc"],
    }
    torch.save(checkpoint, path)
    print(f"  [OK] Checkpoint saved -> {path}  (ROC-AUC: {metrics['roc_auc']:.4f})")


# ===========================================================================
# Optional: matplotlib plots
# ===========================================================================

def plot_training_curves(
    train_losses: List[float],
    accuracies:   List[float],
    roc_aucs:     List[float],
    save_path:    Optional[str] = "training_curves.png",
) -> None:
    """
    Plot training loss, validation accuracy, and ROC-AUC vs epoch.
    Saves to *save_path* if provided; always calls plt.show().
    Silently skips if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib not installed - skipping plots.")
        return

    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("FedMelanoma Training Curves", fontsize=14, fontweight="bold")

    # -- Train Loss -----------------------------------------------------------
    axes[0].plot(epochs, train_losses, marker="o", color="#e05c5c", linewidth=2)
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    # -- Validation Accuracy --------------------------------------------------
    axes[1].plot(
        epochs,
        [a * 100 for a in accuracies],
        marker="s", color="#4c8bf5", linewidth=2,
    )
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)

    # -- ROC-AUC --------------------------------------------------------------
    axes[2].plot(epochs, roc_aucs, marker="^", color="#27ae60", linewidth=2)
    axes[2].set_title("Validation ROC-AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("ROC-AUC")
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n[INFO] Training curves saved -> {save_path}")

    plt.show()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # Fix OpenCV + PyTorch multiprocessing conflict on Linux
    import cv2
    cv2.setNumThreads(0)
    os.environ["OMP_NUM_THREADS"] = "1"

    # -----------------------------------------------------------------------
    # 0. Sanity-check dataset root before doing anything else
    # -----------------------------------------------------------------------
    if not os.path.isdir(DATASET_ROOT):
        raise FileNotFoundError(
            f"DATASET_ROOT does not exist: {DATASET_ROOT}\n"
            f"Update the DATASET_ROOT constant near the top of train.py "
            f"to point at the folder that directly contains your 10 "
            f"hospital subfolders (e.g. CityGeneral_Multispecialty_Hospital)."
        )

    # -----------------------------------------------------------------------
    # 1. Hospital names – unchanged from original
    # -----------------------------------------------------------------------
    hospital_names: List[str] = [
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

    # -----------------------------------------------------------------------
    # 2. Build per-hospital datasets then concatenate
    # -----------------------------------------------------------------------
    train_datasets: List[MelanomaDataset] = []
    test_datasets:  List[MelanomaDataset] = []

    for hospital in hospital_names:
        # CHANGED: base path now built from the DATASET_ROOT constant
        # (absolute path) instead of the hardcoded relative string
        # "hospital_partitions". This is the only structural change vs.
        # the original script — MelanomaDataset itself is untouched.
        base = f"{DATASET_ROOT}/{hospital}"

        train_csv = f"{base}/metadata_train.csv"
        test_csv  = f"{base}/metadata_test.csv"
        train_img = f"{base}/images/Train"
        test_img  = f"{base}/images/Test"

        for path in (train_csv, test_csv):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Expected metadata CSV not found: {path}\n"
                    f"Check that '{hospital}' exists under DATASET_ROOT "
                    f"and contains metadata_train.csv / metadata_test.csv."
                )

        train_datasets.append(MelanomaDataset(train_csv, train_img))
        test_datasets.append(MelanomaDataset(test_csv, test_img))

    train_dataset = ConcatDataset(train_datasets)
    test_dataset  = ConcatDataset(test_datasets)

    # -----------------------------------------------------------------------
    # 3. Collect ALL training labels to compute class weights & statistics
    # -----------------------------------------------------------------------
    train_targets: List[int] = []
    for dataset in train_datasets:
        for label in dataset.df["diagnosis_1"]:
            train_targets.append(
                1 if str(label).lower() == "malignant" else 0
            )

    # Collect test labels for statistics only (NOT used in training)
    test_targets: List[int] = []
    for dataset in test_datasets:
        for label in dataset.df["diagnosis_1"]:
            test_targets.append(
                1 if str(label).lower() == "malignant" else 0
            )

    # Compute inverse-frequency class weights for focal loss
    class_weights: torch.Tensor = compute_class_weights(
        torch.tensor(train_targets)
    ).to(DEVICE)

    # -----------------------------------------------------------------------
    # 4. Print dataset statistics before anything else
    # -----------------------------------------------------------------------
    print_dataset_statistics(train_targets, test_targets, class_weights)

    # -----------------------------------------------------------------------
    # 5. DataLoaders
    # -----------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Device : {DEVICE}")
    print(f"Epochs : {NUM_EPOCHS}   |   Batch size: {BATCH_SIZE}")
    print(f"LR     : {LEARNING_RATE}   |   Weight decay: {WEIGHT_DECAY}\n")

    # -----------------------------------------------------------------------
    # 6. Model, loss, optimiser – unchanged from original
    # -----------------------------------------------------------------------
    model: torch.nn.Module = FedMelanomaModel().to(DEVICE)

    criterion = ClassWeightedFocalLoss(
        gamma_f=FOCAL_GAMMA,
        reduction="mean",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------------------------
    # 7. Tracking variables for early stopping and best-model logging
    # -----------------------------------------------------------------------
    best_roc_auc:   float = 0.0
    best_metrics:   Dict[str, float] = {}
    patience_count: int   = 0

    # History lists for optional plotting
    history_train_loss: List[float] = []
    history_accuracy:   List[float] = []
    history_roc_auc:    List[float] = []
    #initialize training time
    total_training_time = 0.0
    # -----------------------------------------------------------------------
    # 8. Training loop
    # -----------------------------------------------------------------------
    print("ENTERED TRAINING LOOP")

    for epoch in range(1, NUM_EPOCHS + 1):

        print(f"STARTING EPOCH {epoch}")

        epoch_start = time.time()

        avg_train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            class_weights,
            epoch_start
        )
        epoch_time = time.time() - epoch_start

        total_training_time += epoch_time

        avg_epoch_time = total_training_time / epoch

        remaining_epochs = NUM_EPOCHS - epoch

        estimated_remaining = avg_epoch_time * remaining_epochs

        print(
            f"Epoch Time: {epoch_time:.2f}s | "
            f"Average Epoch Time: {avg_epoch_time:.2f}s"
        )

        print(
            f"Estimated Remaining: "
            f"{estimated_remaining/60:.2f} min "
            f"({estimated_remaining/3600:.2f} hrs)"
        )


        estimated_total = epoch_time * NUM_EPOCHS


        metrics, y_true, y_pred, y_prob = evaluate(model, test_loader)

        # -- 8c. Per-epoch console output ------------------------------------
        print_epoch_metrics(epoch, NUM_EPOCHS, avg_train_loss, metrics)

        # Detailed confusion matrix + classification report every epoch
        print_detailed_report(y_true, y_pred, epoch)

        # -- 8d. Track history for plotting ----------------------------------
        history_train_loss.append(avg_train_loss)
        history_accuracy.append(metrics["accuracy"])
        history_roc_auc.append(metrics["roc_auc"])

        # -- 8e. Save best checkpoint based on ROC-AUC -----------------------
        if metrics["roc_auc"] > best_roc_auc:
            best_roc_auc   = metrics["roc_auc"]
            best_metrics   = dict(metrics)
            patience_count = 0  # Reset early-stopping counter

            save_checkpoint(
                CHECKPOINT_PATH, epoch, model, optimizer, metrics
            )
        else:
            patience_count += 1
            print(
                f"  [EarlyStopping] No ROC-AUC improvement "
                f"({patience_count}/{EARLY_STOPPING_PATIENCE})"
            )

        # -- 8f. Early stopping check ----------------------------------------
        if epoch >= MIN_EPOCHS_BEFORE_STOPPING and patience_count >= EARLY_STOPPING_PATIENCE:
            print(
                f"\n[!] Early stopping triggered after epoch {epoch}. "
                f"No ROC-AUC improvement for {EARLY_STOPPING_PATIENCE} "
                f"consecutive epochs."
            )
            break

    # -----------------------------------------------------------------------
    # 9. Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - BEST VALIDATION METRICS")
    print("=" * 60)
    print(f"  Best Accuracy  : {best_metrics.get('accuracy',  0)*100:.2f}%")
    print(f"  Best Precision : {best_metrics.get('precision', 0):.4f}")
    print(f"  Best Recall    : {best_metrics.get('recall',    0):.4f}")
    print(f"  Best F1        : {best_metrics.get('f1',        0):.4f}")
    print(f"  Best ROC-AUC   : {best_metrics.get('roc_auc',  0):.4f}")
    print(f"  Checkpoint     : {CHECKPOINT_PATH}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 10. Optional: plot training curves (requires matplotlib)
    # -----------------------------------------------------------------------
    plot_training_curves(
        history_train_loss,
        history_accuracy,
        history_roc_auc,
        save_path="training_curves.png",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()         

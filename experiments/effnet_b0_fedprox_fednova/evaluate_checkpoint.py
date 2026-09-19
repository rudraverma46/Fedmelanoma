"""
evaluate_checkpoint.py — Run a saved checkpoint over the pooled test set and
dump per-sample predicted probabilities.

The training log stores only per-round summary metrics, which is not enough to
draw ROC / precision-recall / threshold curves — those need the probability
assigned to every individual test image. This script reloads a checkpoint,
runs inference once, and writes the raw scores to CSV so the curve figures can
be generated (and regenerated) without re-running inference.

Usage:
    python evaluate_checkpoint.py \
        --checkpoint full_run_checkpoint.pth \
        --data-root /path/to/hospital_partitions \
        --out test_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
CLIENT_DIR = REPO_ROOT / "client"
sys.path.insert(0, str(CLIENT_DIR))
sys.path.insert(0, str(THIS_DIR))

from model import FedMelanomaModelB0  # noqa: E402
from dataset import MelanomaDataset  # noqa: E402

HOSPITALS = [
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


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="full_run_checkpoint.pth")
    p.add_argument("--data-root", default="hospital_partitions")
    p.add_argument("--out", default="test_predictions.csv")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"[Eval] Checkpoint from round {ckpt['round']}, "
          f"roc_auc={ckpt['metrics']['roc_auc']:.4f}")

    model = FedMelanomaModelB0()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # Pooled test set, in the same hospital order used during training, so the
    # row order here lines up with the per-hospital record below.
    datasets, hosp_of_row = [], []
    for h in HOSPITALS:
        ds = MelanomaDataset(str(root / h / "metadata_test.csv"),
                             str(root / h / "images" / "Test"))
        datasets.append(ds)
        hosp_of_row.extend([h] * len(ds))
        print(f"  {h}: {len(ds)} test images")

    loader = DataLoader(ConcatDataset(datasets), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    probs, labels = [], []
    total = sum(len(d) for d in datasets)
    with torch.no_grad():
        for i, (images, meta, y) in enumerate(loader):
            logits = model(images.to(device), meta.to(device))
            probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())
            if i % 20 == 0:
                print(f"  inference {len(probs)}/{total}")

    df = pd.DataFrame({
        "hospital": hosp_of_row,
        "label": labels,
        "prob_malignant": probs,
    })
    df.to_csv(args.out, index=False)
    print(f"[Eval] Wrote {len(df)} predictions to {args.out}")
    print(f"       malignant: {int(df['label'].sum())}  benign: {int((df['label']==0).sum())}")


if __name__ == "__main__":
    main()

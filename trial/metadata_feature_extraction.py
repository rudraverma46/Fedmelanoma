import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# ==========================================================
# SETTINGS
# ==========================================================

ROOT_DIR = r"E:\Hospital_partitions"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# ==========================================================
# MLP
# ==========================================================

class MetadataMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(13, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, x):

        return self.network(x)

# ==========================================================
# ONE HOT ENCODING
# ==========================================================

def create_metadata_vectors(df):

    metadata_vectors = []

    for _, row in df.iterrows():

        vec = np.zeros(13, dtype=np.float32)

        # --------------------------------------------------
        # FITZPATRICK
        # --------------------------------------------------

        skin = str(
            row["fitzpatrick_skin_type"]
        ).strip()

        skin_map = {
            "I":0,
            "II":1,
            "III":2,
            "IV":3,
            "V":4,
            "VI":5
        }

        if skin in skin_map:
            vec[skin_map[skin]] = 1

        # --------------------------------------------------
        # IMAGE TYPE
        # --------------------------------------------------

        image = str(
            row["image_type"]
        ).strip().lower()

        image_map = {
            "dermoscopic":0,
            "clinical_closeup":1,
            "clinical_overview":2
        }

        if image in image_map:
            vec[6 + image_map[image]] = 1

        # --------------------------------------------------
        # CONFIRM TYPE
        # --------------------------------------------------

        confirm = str(
            row["diagnosis_confirm_type"]
        ).strip().lower()

        confirm_map = {
            "histopathology":0,
            "expert_consensus":1,
            "follow_up":2,
            "single_contributor":3
        }

        if confirm in confirm_map:
            vec[9 + confirm_map[confirm]] = 1

        metadata_vectors.append(vec)

    return np.array(metadata_vectors)

# ==========================================================
# MODEL
# ==========================================================

mlp = MetadataMLP().to(device)
mlp.eval()

# ==========================================================
# PROCESS ALL HOSPITALS
# ==========================================================

hospital_folders = [

    d for d in os.listdir(ROOT_DIR)

    if os.path.isdir(
        os.path.join(ROOT_DIR, d)
    )
]

for hospital in hospital_folders:

    hospital_path = os.path.join(
        ROOT_DIR,
        hospital
    )

    train_csv = os.path.join(
        hospital_path,
        "metadata_train.csv"
    )

    test_csv = os.path.join(
        hospital_path,
        "metadata_test.csv"
    )

    if not os.path.exists(train_csv):
        continue

    if not os.path.exists(test_csv):
        continue

    print("\n" + "="*60)
    print(f"Processing {hospital}")
    print("="*60)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_vectors = create_metadata_vectors(
        train_df
    )

    test_vectors = create_metadata_vectors(
        test_df
    )

    with torch.no_grad():

        train_meta_features = (
            mlp(
                torch.tensor(
                    train_vectors,
                    dtype=torch.float32
                ).to(device)
            )
            .cpu()
            .numpy()
        )

        test_meta_features = (
            mlp(
                torch.tensor(
                    test_vectors,
                    dtype=torch.float32
                ).to(device)
            )
            .cpu()
            .numpy()
        )

    np.save(
        os.path.join(
            hospital_path,
            "train_metadata_features.npy"
        ),
        train_meta_features
    )

    np.save(
        os.path.join(
            hospital_path,
            "test_metadata_features.npy"
        ),
        test_meta_features
    )

    print(
        "Train Metadata:",
        train_meta_features.shape
    )

    print(
        "Test Metadata:",
        test_meta_features.shape
    )

print("\nALL HOSPITALS COMPLETED")
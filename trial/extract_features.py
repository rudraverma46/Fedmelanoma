import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# ==========================================================
# SETTINGS
# ==========================================================

ROOT_DIR = r"E:\Hospital_partitions"
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================================
# EFFICIENTNET-B3
# ==========================================================

weights = EfficientNet_B3_Weights.IMAGENET1K_V1

model = efficientnet_b3(weights=weights)

# Remove classifier
model.classifier = torch.nn.Identity()

model = model.to(device)
model.eval()

# ==========================================================
# TRANSFORM
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

@torch.no_grad()
def extract_folder_features(folder_path):

    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ]

    features = []
    names = []

    for start in tqdm(
        range(0, len(image_files), BATCH_SIZE),
        desc=os.path.basename(folder_path)
    ):

        batch_files = image_files[start:start+BATCH_SIZE]

        batch_imgs = []

        for img_name in batch_files:

            img_path = os.path.join(folder_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img = transform(img)

                batch_imgs.append(img)
                names.append(img_name)

            except Exception as e:
                print(f"Error: {img_path}")
                print(e)

        if len(batch_imgs) == 0:
            continue

        batch_tensor = torch.stack(batch_imgs).to(device)

        batch_features = model(batch_tensor)

        batch_features = (
            batch_features
            .cpu()
            .numpy()
        )

        features.append(batch_features)

    if len(features) == 0:
        return np.array([]), np.array([])

    features = np.vstack(features)

    return np.array(names), features


# ==========================================================
# PROCESS ALL HOSPITALS
# ==========================================================

hospital_folders = [
    d for d in os.listdir(ROOT_DIR)
    if os.path.isdir(os.path.join(ROOT_DIR, d))
]

for hospital in hospital_folders:

    hospital_path = os.path.join(ROOT_DIR, hospital)

    train_dir = os.path.join(
        hospital_path,
        "images",
        "Train"
    )

    test_dir = os.path.join(
        hospital_path,
        "images",
        "Test"
    )

    if not os.path.exists(train_dir):
        continue

    if not os.path.exists(test_dir):
        continue

    print("\n" + "="*60)
    print(f"Processing {hospital}")
    print("="*60)

    train_names, train_features = extract_folder_features(
        train_dir
    )

    test_names, test_features = extract_folder_features(
        test_dir
    )

    np.save(
        os.path.join(hospital_path, "train_features.npy"),
        train_features
    )

    np.save(
        os.path.join(hospital_path, "train_image_names.npy"),
        train_names
    )

    np.save(
        os.path.join(hospital_path, "test_features.npy"),
        test_features
    )

    np.save(
        os.path.join(hospital_path, "test_image_names.npy"),
        test_names
    )

    print(
        f"Train shape: {train_features.shape}"
    )

    print(
        f"Test shape: {test_features.shape}"
    )

print("\nALL HOSPITALS COMPLETED")
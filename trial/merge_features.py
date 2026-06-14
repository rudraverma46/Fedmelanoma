import os
import numpy as np

ROOT_DIR = r"E:\Hospital_partitions"

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

    train_img = os.path.join(
        hospital_path,
        "train_features.npy"
    )

    test_img = os.path.join(
        hospital_path,
        "test_features.npy"
    )

    train_meta = os.path.join(
        hospital_path,
        "train_metadata_features.npy"
    )

    test_meta = os.path.join(
        hospital_path,
        "test_metadata_features.npy"
    )

    if not all([
        os.path.exists(train_img),
        os.path.exists(test_img),
        os.path.exists(train_meta),
        os.path.exists(test_meta)
    ]):
        continue

    print("\n" + "="*60)
    print(f"Processing {hospital}")
    print("="*60)

    train_features = np.load(
        train_img
    )

    test_features = np.load(
        test_img
    )

    train_meta_features = np.load(
        train_meta
    )

    test_meta_features = np.load(
        test_meta
    )

    print(
        "Train Image:",
        train_features.shape
    )

    print(
        "Train Metadata:",
        train_meta_features.shape
    )

    merged_train = np.concatenate(
        [
            train_features,
            train_meta_features
        ],
        axis=1
    )

    merged_test = np.concatenate(
        [
            test_features,
            test_meta_features
        ],
        axis=1
    )

    np.save(
        os.path.join(
            hospital_path,
            "train_combined_features.npy"
        ),
        merged_train
    )

    np.save(
        os.path.join(
            hospital_path,
            "test_combined_features.npy"
        ),
        merged_test
    )

    print(
        "Merged Train:",
        merged_train.shape
    )

    print(
        "Merged Test:",
        merged_test.shape
    )

print("\nALL HOSPITALS COMPLETED")
from pathlib import Path
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split


def create_train_test_split(
    hospitals,
    output_dir,
    test_size=0.20,
    random_state=42
):

    for h in hospitals:

        hospital_folder = output_dir / h["name"]

        metadata_file = hospital_folder / "metadata.csv"

        if not metadata_file.exists():
            continue

        print(f"[split] {h['name']}")

        df = pd.read_csv(metadata_file)

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["diagnosis_1"]
        )

        train_dir = hospital_folder / "images" / "Train"
        test_dir = hospital_folder / "images" / "Test"

        train_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        test_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        image_dir = hospital_folder / "images"

        for _, row in train_df.iterrows():

            img_id = str(row["isic_id"])

            for ext in [
                ".jpg",
                ".jpeg",
                ".png"
            ]:

                src = image_dir / f"{img_id}{ext}"

                if src.exists():
                    shutil.copy2(
                        src,
                        train_dir / src.name
                    )
                    break

        for _, row in test_df.iterrows():

            img_id = str(row["isic_id"])

            for ext in [
                ".jpg",
                ".jpeg",
                ".png"
            ]:

                src = image_dir / f"{img_id}{ext}"

                if src.exists():
                    shutil.copy2(
                        src,
                        test_dir / src.name
                    )
                    break

        train_df.to_csv(
            hospital_folder / "metadata_train.csv",
            index=False
        )

        test_df.to_csv(
            hospital_folder / "metadata_test.csv",
            index=False
        )

        print(
            f"[split] Train={len(train_df)} "
            f"Test={len(test_df)}"
        )
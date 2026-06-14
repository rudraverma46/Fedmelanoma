from pathlib import Path
import pandas as pd

def create_distribution_files(hospitals, output_dir):

    columns = [
        "diagnosis_1",
        "fitzpatrick_skin_type",
        "diagnosis_confirm_type"
    ]

    for h in hospitals:

        hospital_folder = output_dir / h["name"]

        metadata_file = hospital_folder / "metadata.csv"
        train_file = hospital_folder / "metadata_train.csv"
        test_file = hospital_folder / "metadata_test.csv"

        if not metadata_file.exists():
            continue

        if not train_file.exists():
            print(
                f"[distribution] Missing train file for "
                f"{h['name']}"
            )
            continue

        if not test_file.exists():
            print(
                f"[distribution] Missing test file for "
                f"{h['name']}"
            )
            continue

        df = pd.read_csv(metadata_file)
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        total_images = len(df)

        for col in columns:

            if col not in df.columns:
                continue

            overall = (
                df[col]
                .fillna("Missing")
                .astype(str)
                .value_counts()
            )

            train_counts = (
                train_df[col]
                .fillna("Missing")
                .astype(str)
                .value_counts()
            )

            test_counts = (
                test_df[col]
                .fillna("Missing")
                .astype(str)
                .value_counts()
            )

            categories = sorted(
                set(overall.index)
                | set(train_counts.index)
                | set(test_counts.index)
            )

            rows = []

            for cat in categories:

                total_count = int(
                    overall.get(cat, 0)
                )

                train_count = int(
                    train_counts.get(cat, 0)
                )

                test_count = int(
                    test_counts.get(cat, 0)
                )

                percentage = round(
                    total_count
                    / total_images
                    * 100,
                    2
                )

                rows.append({
                    col: cat,
                    "Image_Count": total_count,
                    "Percentage": percentage,
                    "Train_Count": train_count,
                    "Test_Count": test_count
                })

            summary = pd.DataFrame(rows)

            output_file = (
                hospital_folder
                / f"{col}_distribution.csv"
            )

            summary.to_csv(
                output_file,
                index=False
            )

            print(
                f"[distribution] Saved "
                f"{output_file}"
            )
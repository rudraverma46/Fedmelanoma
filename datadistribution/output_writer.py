"""
output_writer.py
================
Creates per-hospital folders, copies images, writes per-hospital
metadata.csv, and the global missing_images.csv.
"""

import shutil
from pathlib import Path
import pandas as pd
from config import OUTPUT_DIR
from metadata_loader import find_image_file


def create_output_folders(hospitals: list[dict], output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for h in hospitals:
        (output_dir / h["name"] / "images").mkdir(parents=True, exist_ok=True)
    print(f"[output_writer] Created {len(hospitals)} hospital folders under {output_dir}")


def copy_images_to_hospitals(df: pd.DataFrame,
                             hospitals: list[dict],
                             id_col: str,
                             input_dir: Path,
                             output_dir: Path = OUTPUT_DIR) -> tuple[dict, list[dict]]:
    """
    Copies images per assigned hospital. Returns (per_hospital_counts, missing_records).
    """
    per_hospital_copied  = {h["name"]: 0 for h in hospitals}
    per_hospital_missing = {h["name"]: 0 for h in hospitals}
    missing_records = []

    for _, row in df.iterrows():
        hosp = row["assigned_hospital"]
        image_id = row[id_col]
        src = find_image_file(image_id, images_dir=input_dir)
        if src is None:
            per_hospital_missing[hosp] += 1
            missing_records.append({
                "image_id": image_id,
                "assigned_hospital": hosp,
            })
            continue
        dst_dir = output_dir / hosp / "images"
        dst = dst_dir / src.name
        if not dst.exists():
            try:
                shutil.copy2(src, dst)
                per_hospital_copied[hosp] += 1
            except Exception as e:
                per_hospital_missing[hosp] += 1
                missing_records.append({
                    "image_id": image_id,
                    "assigned_hospital": hosp,
                    "error": str(e),
                })
        else:
            per_hospital_copied[hosp] += 1

    print(f"[output_writer] Copy complete. Missing/failed: {len(missing_records)}")
    return {"copied": per_hospital_copied, "missing": per_hospital_missing}, missing_records


def save_hospital_metadata(df,
                           hospitals,
                           output_dir=OUTPUT_DIR):

    required_cols = [
        "isic_id",
        "diagnosis_1",
        "fitzpatrick_skin_type",
        "image_type",
        "diagnosis_confirm_type"
    ]

    existing_cols = [
        c for c in required_cols
        if c in df.columns
    ]

    for h in hospitals:

        sub = df[
            df["assigned_hospital"] == h["name"]
        ][existing_cols].copy()

        out_csv = (
            output_dir /
            h["name"] /
            "metadata.csv"
        )

        sub.to_csv(
            out_csv,
            index=False
        )

        print(
            f"  - Saved {out_csv} "
            f"({len(sub)} rows)"
        )
        print(f"  - Saved {out_csv} ({len(sub)} rows)")


def save_missing_images_report(missing_records: list[dict], output_dir: Path = OUTPUT_DIR) -> Path:
    out = output_dir / "missing_images.csv"
    if missing_records:
        pd.DataFrame(missing_records).to_csv(out, index=False)
    else:
        pd.DataFrame(columns=["image_id", "assigned_hospital"]).to_csv(out, index=False)
    print(f"[output_writer] Missing images report -> {out}")
    return out
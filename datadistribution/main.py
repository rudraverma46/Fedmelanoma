"""
main.py
=======
End-to-end pipeline:
  1. Load metadata + (optional) blueprint
  2. Detect columns
  3. Compute f(m_i) = [f_phi, f_tau, f_lambda]
  4. Create hospital archetypes
  5. Generate w_k from distributions + blueprint + archetype
  6. Probabilistically assign samples (softmax)
  7. [NEW] Adjust malignant-range across hospitals (optional, config-gated)
  8. Copy images, save per-hospital metadata
  9. Write hospital_description.md, process_explanation.md, summaries
"""

from pathlib import Path
import pandas as pd

from config import (
    INPUT_DIR, OUTPUT_DIR, RANDOM_SEED, TAU_SOFT,
    MAX_HOSPITAL_CAPACITY_FRACTION,
    ENABLE_MALIGNANT_RANGE_ADJUSTMENT,          # NEW
)
from metadata_loader import (
    find_metadata_csv, find_blueprint_file, load_metadata, detect_all_columns,
)
from feature_scoring import build_sample_vector
from hospital_profiles import create_hospital_archetypes
from wk_generator import (
    analyze_metadata_distributions, generate_wk_values, validate_wk_values,
)
from assignment import assign_hospitals
from output_writer import (
    create_output_folders, copy_images_to_hospitals,
    save_hospital_metadata, save_missing_images_report,
)
from markdown_writer import write_hospital_description_md, write_process_explanation_md
from summary_writer import create_partition_summary, save_partition_summary
from hospital_distribution_writer import create_distribution_files


def main():
    print("=" * 70)
    print("ISIC Federated Learning -- Non-IID Hospital Partitioning")
    print("=" * 70)

    # 1. Locate inputs
    csv_path = find_metadata_csv(INPUT_DIR)
    if csv_path is None:
        raise FileNotFoundError(f"No metadata CSV found in {INPUT_DIR}")
    blueprint_path = find_blueprint_file(INPUT_DIR)
    blueprint_text = blueprint_path.read_text(encoding="utf-8", errors="ignore") if blueprint_path else None
    print(f"[main] Metadata CSV : {csv_path}")
    print(f"[main] Blueprint    : {blueprint_path if blueprint_path else 'NOT FOUND (continuing without)'}")

    # 2. Load metadata + detect columns
    df = load_metadata(csv_path)
    cols = detect_all_columns(df)
    print(f"[main] Detected columns: {cols}")
    if cols["id"] is None:
        raise ValueError("Could not detect ISIC ID / image filename column. Edit config.py to add candidates.")

    # 3. Compute f(m_i)
    img_type_map = None
    print("[main] Image type scoring: tier-based formula from feature_scoring.py")

    f_phi, f_tau, f_lambda = [], [], []
    for _, row in df.iterrows():
        a, b, c = build_sample_vector(row, cols, img_type_map)
        f_phi.append(a); f_tau.append(b); f_lambda.append(c)
    df["f_phi"]    = f_phi
    df["f_tau"]    = f_tau
    df["f_lambda"] = f_lambda
    df["f_mi_vector"] = df.apply(
        lambda r: f"[{r['f_phi']:.4f}, {r['f_tau']:.4f}, {r['f_lambda']:.4f}]", axis=1
    )

    # 4. Hospital archetypes
    archetypes = create_hospital_archetypes()

    # 5. Distributions + w_k generation
    distributions = analyze_metadata_distributions(df, cols, img_type_map)
    hospitals = generate_wk_values(archetypes, distributions, blueprint_text, RANDOM_SEED)
    validate_wk_values(hospitals)

    # 6. Assignment (original softmax — unchanged)
    df = assign_hospitals(
        df,
        hospitals,
        tau_soft=TAU_SOFT,
        seed=RANDOM_SEED,
        max_capacity_fraction=MAX_HOSPITAL_CAPACITY_FRACTION,
    )

    # ── 7. NEW: Malignant-range adjustment ─────────────────────────────────
    # Runs AFTER the original softmax assignment, BEFORE any output is written.
    # Controlled by ENABLE_MALIGNANT_RANGE_ADJUSTMENT in config.py.
    if ENABLE_MALIGNANT_RANGE_ADJUSTMENT:
        from malignant_range_adjuster import adjust_malignant_range
        df = adjust_malignant_range(df, hospitals, OUTPUT_DIR)
    # ──────────────────────────────────────────────────────────────────────

    # 8. Output folders + image copy + per-hospital metadata
    create_output_folders(hospitals, OUTPUT_DIR)
    counts, missing_records = copy_images_to_hospitals(
        df, hospitals, id_col=cols["id"], input_dir=INPUT_DIR, output_dir=OUTPUT_DIR,
    )
    save_hospital_metadata(df, hospitals, OUTPUT_DIR)
    from train_test_split import create_train_test_split

    create_train_test_split(
    hospitals,
    OUTPUT_DIR,
    test_size=0.20,
    random_state=42
    )
    create_distribution_files(hospitals, OUTPUT_DIR)
    save_missing_images_report(missing_records, OUTPUT_DIR)

    # 9. Markdown + summary
    for h in hospitals:
        sub = df[df["assigned_hospital"] == h["name"]]
        write_hospital_description_md(
            h, sub, cols,
            copied=counts["copied"][h["name"]],
            missing=counts["missing"][h["name"]],
            output_dir=OUTPUT_DIR,
        )
    write_process_explanation_md(hospitals, cols, distributions, OUTPUT_DIR)

    summary_df = create_partition_summary(df, hospitals, cols, counts["copied"], counts["missing"])
    save_partition_summary(summary_df, OUTPUT_DIR)

    print("\n[main] DONE.")
    print(f"[main] Output -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

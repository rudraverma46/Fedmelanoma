"""
config.py
=========
Central configuration for the Non-IID hospital partitioning pipeline.
Edit values here if your paths / column names differ.
"""

from pathlib import Path

# -------- Paths --------
INPUT_DIR  = Path(r"D:\capstone\Images")
OUTPUT_DIR = Path(r"D:\capstone\Images\hospital_partitions")

# -------- Reproducibility --------
RANDOM_SEED = 42

# -------- Softmax temperature for probabilistic assignment --------
# Smaller -> stronger Non-IID, Larger -> closer to IID/random.
TAU_SOFT = 0.5

# -------- Capacity control --------
# Maximum fraction of total samples a single hospital may receive.
# Helps avoid one hospital dominating the partition.
MAX_HOSPITAL_CAPACITY_FRACTION = 0.25

# -------- Image extensions to look for --------
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

# -------- Candidate column names (we auto-detect) --------
ID_COLUMN_CANDIDATES         = ["isic_id", "image_id", "image", "image_name", "name", "id"]
FITZPATRICK_COLUMN_CANDIDATES= ["fitzpatrick_skin_type", "fitzpatrick", "skin_type"]
IMAGE_TYPE_COLUMN_CANDIDATES = ["image_type", "imagetype", "img_type"]
CONFIRM_COLUMN_CANDIDATES    = ["diagnosis_confirm_type", "confirm_type", "diagnosis_confirmation"]

# Number of hospitals (fixed for this project)
NUM_HOSPITALS = 10

# ════════════════════════════════════════════════════════════════════════
# NEW: Malignant-range adjustment settings
# ════════════════════════════════════════════════════════════════════════

# Set to False to skip the adjustment and keep the original softmax result.
ENABLE_MALIGNANT_RANGE_ADJUSTMENT = False

# Optional: Override per-hospital target malignant percentages (0–100).
# If a hospital name is not listed here, malignant_range_adjuster.py
# will derive the target automatically from the hospital type.
# Example:
#   TARGET_MALIGNANT_PERCENTAGES = {
#       "NorthCare_Tertiary_Melanoma_Institute": 70.0,
#       "GoldenSun_Public_Health_Screening_Camp": 5.0,
#   }
TARGET_MALIGNANT_PERCENTAGES: dict = {}

# Adjustment strength: 1.0 = try to reach exact target; 0.0 = no adjustment.
# Values between 0 and 1 partially preserve the original softmax assignment
# while still widening the malignant range.
MALIGNANT_ADJUSTMENT_STRENGTH: float = 1.0

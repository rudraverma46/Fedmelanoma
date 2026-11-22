"""Copies the hospital partitions without the diagnosis_confirm_type column, which leaks the label.

    python -m ablation.make_noconfirm_dataset SRC DST

CSV files are rewritten without that column (every other value is kept as written), files that only
describe confirmation types are left out, and image folders are symlinked rather than copied. SRC is
only read.
"""
from __future__ import annotations

import csv
import os
import shutil
import sys

COLUMN = "diagnosis_confirm_type"


def strip_column(src: str, dst: str) -> bool:
    with open(src, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows or COLUMN not in rows[0]:
        shutil.copy2(src, dst)
        return False
    k = rows[0].index(COLUMN)
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, lineterminator="\n").writerows(r[:k] + r[k + 1:] for r in rows)
    with open(dst, newline="", encoding="utf-8") as fh:
        check = list(csv.reader(fh))
    if COLUMN in check[0] or len(check) != len(rows):
        raise RuntimeError(f"rewrite of {src} failed")
    return True


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = (os.path.abspath(p) for p in sys.argv[1:3])
    if os.path.exists(dst):
        raise SystemExit(f"{dst} already exists; remove it or choose another path")
    counts = {"hospitals": 0, "csv_stripped": 0, "csv_copied": 0, "skipped": 0, "linked": 0, "copied": 0}
    for hospital in sorted(os.listdir(src)):
        s = os.path.join(src, hospital)
        if not os.path.isdir(s):
            continue
        d = os.path.join(dst, hospital)
        os.makedirs(d)
        counts["hospitals"] += 1
        for name in sorted(os.listdir(s)):
            sp, dp = os.path.join(s, name), os.path.join(d, name)
            if os.path.isdir(sp):
                os.symlink(sp, dp)
                counts["linked"] += 1
            elif COLUMN in name:
                counts["skipped"] += 1
            elif name.endswith(".csv"):
                counts["csv_stripped" if strip_column(sp, dp) else "csv_copied"] += 1
            else:
                shutil.copy2(sp, dp)
                counts["copied"] += 1
    print(f"{dst}: {counts}")


if __name__ == "__main__":
    main()

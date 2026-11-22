"""How much could the client-to-hospital mapping (difference D3) change what the model learns from?

The schedule depends only on client_id, so it is the same under any mapping. What the mapping
changes is whose data sits behind each client_id. Under the reference, client_id k carries
hospital k's n, D and tier, but its update comes from hospital pi(k), for a random pi.

For each random pi this computes, without training:
  - each real hospital's share of total aggregation weight over the 50 planned rounds
  - the malignant rate of the data the global model is pulled towards (weight-averaged)

This measures how different the training signal is. It does NOT measure the effect on ROC-AUC,
which only training could show. Pure arithmetic on planned_schedule.json; needs numpy only.
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERMUTATIONS = 10000


def main() -> None:
    with open(os.path.join(ROOT, "results", "schedule", "planned_schedule.json"), encoding="utf-8") as fh:
        plan = json.load(fh)
    with open(os.path.join(ROOT, "documentation", "plan_hospital_facts.json"), encoding="utf-8") as fh:
        facts = json.load(fh)
    k = len(facts)
    weight = np.zeros(k)  # total c_i per client_id over all rounds
    for row in plan["schedule"]:
        for cid, c in row["weights"].items():
            weight[int(cid)] += c
    share = weight / weight.sum()
    rate = np.array([h["mal_train"] / h["n_train"] for h in facts])

    rng = np.random.default_rng(0)
    shares = np.zeros((PERMUTATIONS, k))
    exposure = np.zeros(PERMUTATIONS)
    moved = np.zeros(PERMUTATIONS)
    for i in range(PERMUTATIONS):
        pi = rng.permutation(k)            # client_id j's update comes from hospital pi[j]
        shares[i, pi] = share
        exposure[i] = (shares[i] * rate).sum()
        moved[i] = 0.5 * np.abs(shares[i] - share).sum()

    out = {
        "permutations": PERMUTATIONS,
        "note": "Arithmetic on the planned schedule. Not a measurement of model performance.",
        "fixed_mapping": {
            "hospital_weight_share": {facts[j]["hospital"]: float(share[j]) for j in range(k)},
            "malignant_rate_of_weighted_data": float((share * rate).sum()),
        },
        "random_mapping": {
            "hospital_weight_share_p5_p95": {facts[j]["hospital"]: [float(np.percentile(shares[:, j], 5)), float(np.percentile(shares[:, j], 95))] for j in range(k)},
            "malignant_rate_mean": float(exposure.mean()),
            "malignant_rate_p5_p95": [float(np.percentile(exposure, 5)), float(np.percentile(exposure, 95))],
            "fraction_of_weight_moved_to_other_hospitals_mean": float(moved.mean()),
            "fraction_of_weight_moved_p5_p95": [float(np.percentile(moved, 5)), float(np.percentile(moved, 95))],
        },
    }
    path = os.path.join(ROOT, "results", "schedule", "mapping_sensitivity.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

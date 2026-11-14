"""Checks the blueprint metrics on a synthetic finished run.

    python -m pytest ablation/test_blueprint_metrics.py -q
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ablation.analysis import blueprint_metrics as bm  # noqa: E402


def test_ece_and_temperature_recover_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-z))).astype(float)
    over = bm.scale(1 / (1 + np.exp(-z)), 1 / 3)            # logits x3: overconfident
    assert bm.ece(y, 1 / (1 + np.exp(-z))) < 0.02 < bm.ece(y, over)
    assert bm.fit_temperature(y, over) == pytest.approx(3.0, rel=0.1)


def test_gini():
    assert bm.gini(np.full(10, 18)) == pytest.approx(0.0)
    assert bm.gini(np.array([0] * 9 + [10])) == pytest.approx(0.9)


def test_run_metrics(tmp_path):
    run = tmp_path / "A8_synchronous" / "seed_42"
    (run / "logs").mkdir(parents=True)
    (run / "final_round" / "results" / "metrics").mkdir(parents=True)
    pd.DataFrame({"round": range(1, 31), "eval_roc_auc": [0.5] * 10 + [0.7] * 10 + [0.8] * 10}).to_csv(
        run / "metrics.csv", index=False)
    pd.DataFrame({"subgroup": ["I", "II", "V"], "sensitivity": [0.9, 0.6, np.nan]}).to_csv(
        run / "final_round" / "results" / "metrics" / "skin_type_metrics.csv", index=False)
    recs = []
    for rnd, (lat, late, failed) in enumerate([((2.0, 3.0), [], []), ((2.0, 50.0), [5], []),
                                               ((4.0,), [], [7])], start=1):
        ids = [0, 5] if rnd < 3 else [0, 7]
        recs.append({"event": "round_summary", "round": rnd, "adaptive_window_w_t": 10.0, "selected_set": ids,
                     "straggler_set": late, "failed_set": failed})
        for cid, l in zip(ids, list(lat) + [None]):
            cls = "failed" if cid in failed else ("straggler" if cid in late else "on_time")
            recs.append({"event": "client_round_outcome", "round": rnd, "client_id": cid, "selected": True,
                         "response_classification": cls, "latency_seconds": l})
    with open(run / "logs" / "stragglers.jsonl", "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in recs))

    assert bm.late_round_gain(str(run)) == pytest.approx(0.1)
    assert bm.fairness_gap(str(run)) == pytest.approx(0.3)
    sel = bm.selection_and_time(str(run))
    assert sel["min_selections"] == 0 and sel["never_selected"] == 7
    assert sel["sim_hours_window"] * 3600 == pytest.approx(3.0 + 10.0 + 10.0)
    assert sel["sim_hours_sync"] * 3600 == pytest.approx(3.0 + 50.0 + bm.SYNC_TIMEOUT_S)

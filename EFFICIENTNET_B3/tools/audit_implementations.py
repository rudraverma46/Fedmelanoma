"""Audits three implementations from their source: the reference, ResNet-50, and this experiment.

Each audit item names the file and symbol that implement it. The tool confirms the symbol exists in
that file and compares file hashes, so a status can only say "identical" if the bytes are. Writes
documentation/AUDIT_REPORT.md and results/audit/audit.json.

Usage: python tools/audit_implementations.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLONE = os.path.abspath(os.path.join(ROOT, "..", "..", "FedMelanoma_repo"))
RESNET = os.path.abspath(os.path.join(ROOT, "..", "RESNET 50_ fed novelty", "RESNET50_EFFICIENTNET_B3_EQUIVALENT"))
COMMIT = "1351f63b37c022a252c18960098eb388bb2f723a"

# (item, file, text that must appear in the file, what the code does)
ITEMS = [
    ("Model architecture", "models/model.py", "class FedMelanomaModel", "Backbone, metadata branch, fusion, head"),
    ("Image preprocessing", "client/dataset.py", "cv2.resize(image, (224, 224))", "RGB, 224x224, ImageNet normalisation, no augmentation"),
    ("Metadata preprocessing", "client/dataset.py", "encode_metadata", "13-dimensional one-hot vector"),
    ("Metadata MLP", "models/model.py", "class MetadataMLP", "13 -> 64 (BatchNorm, ReLU) -> 32 (ReLU)"),
    ("Feature fusion", "models/model.py", "torch.cat", "[image ; metadata]"),
    ("Classification head", "models/model.py", "self.classifier", "Dropout 0.3, then Linear(F + 32, 2)"),
    ("Focal loss", "client/focal_loss.py", "class ClassWeightedFocalLoss", "Class-weighted focal, gamma 2, per-hospital weights"),
    ("Client selection algorithm", "server/selector.py", "boltzmann_selection", "Boltzmann sampling without replacement, tau 0.20"),
    ("Diversity calculation", "emd.py", "compute_diversity_score", "OT distance, weights 0.40 / 0.35 / 0.25"),
    ("Reliability calculation", "server/selector.py", "update_client_state", "R_i moving average, rho 0.85"),
    ("Exposure-aware selection", "server/selector.py", "compute_score", "R_i(1 + gamma D_i / (1 + lambda N_eff))"),
    ("Aggregation equations", "server/aggregation.py", "compute_aggregation_weights", "(n_i/n_A) delta^s_i (1 + gamma D_i), normalised"),
    ("Staleness handling", "server/aggregation.py", "DELTA", "delta 0.90 discount"),
    ("Straggler simulation", "server/straggler.py", "simulate_response", "Failure and latency drawn after local fit"),
    ("Client latency distributions", "server/straggler.py", "BETA_PARAMS", "Beta completion and LogNormal latency by tier"),
    ("Adaptive window calculation", "server/straggler.py", "_update_window", "max(median + 2.5 MAD, 2.0 s), last 10 latencies"),
    ("Quorum calculation and behaviour", "server/straggler.py", "manage_window", "ceil(0.7 C), logged only, never blocks"),
    ("Late-update buffering", "server/straggler.py", "_drain_buffer", "s = 1, drained next round, S_max 3"),
    ("Local optimizer", "client/local_trainer.py", "build_optimizer", "AdamW, new each fit; no scheduler, clipping or AMP"),
    ("Learning rate and weight decay", "client/client.py", "DEFAULT_LEARNING_RATE", "1e-4 and 1e-4"),
    ("Batch size", "client/client.py", "DEFAULT_BATCH_SIZE", "20"),
    ("Local epochs", "configs/experiment_config.yaml", "local_epochs: 5", "5 (reference: pyproject.toml)"),
    ("Federated rounds", "configs/experiment_config.yaml", "num_server_rounds: 50", "50 (reference: pyproject.toml)"),
    ("Client participation", "configs/experiment_config.yaml", "clients_per_round: 6", "6 of 10, buffered clients excluded"),
    ("Dataset partitions", "hospital_registry.py", "build_hospital_registry", "Same read-only partition folder"),
    ("Random seeds", "server/server.py", "seed", "Seed 42 in the reference; torch seeds added by runtime (D7)"),
    ("Checkpointing", "monitoring/checkpointing.py", "save_global_checkpoint", "Every round; best by weighted ROC-AUC"),
    ("Evaluation methodology", "evaluate.py", "compute_metrics", "Pooled test set, threshold 0.5, selected clients per round"),
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return fh.read()


def reference(rel):
    done = subprocess.run(["git", "-C", CLONE, "show", f"{COMMIT}:{rel}"], capture_output=True)
    return done.stdout if done.returncode == 0 else None


def compare(a, b):
    if a is None or b is None:
        return "not in both"
    return "identical" if sha(a) == sha(b) else "differs"


def main() -> None:
    rows = []
    for item, rel, needle, meaning in ITEMS:
        ours, resnet, ref = read(os.path.join(ROOT, rel)), read(os.path.join(RESNET, rel)), reference(rel)
        assert ours is not None and needle.encode() in ours, (item, rel, needle)
        rows.append({"item": item, "file": rel, "symbol": needle, "behaviour": meaning, "this_vs_reference": compare(ours, ref), "this_vs_resnet50": compare(ours, resnet)})

    files = []
    for top in ("client", "server", "monitoring", "models", "runtime", "evaluation", "tools"):
        for folder, _, names in os.walk(os.path.join(ROOT, top)):
            for name in names:
                if name.endswith(".py"):
                    files.append(os.path.relpath(os.path.join(folder, name), ROOT).replace(os.sep, "/"))
    files += ["emd.py", "hospital_registry.py", "evaluate.py"]
    table = [{"file": rel, "vs_reference": compare(read(os.path.join(ROOT, rel)), reference(rel)), "vs_resnet50": compare(read(os.path.join(ROOT, rel)), read(os.path.join(RESNET, rel)))} for rel in sorted(files)]

    os.makedirs(os.path.join(ROOT, "results", "audit"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "audit", "audit.json"), "w", encoding="utf-8") as fh:
        json.dump({"items": rows, "files": table}, fh, indent=1)

    lines = ["# Implementation audit", "",
             f"Compared from source on three trees: the reference (commit `{COMMIT[:7]}`), the ResNet-50 experiment, and this experiment. "
             "\"identical\" means the file's bytes match. Generated by `tools/audit_implementations.py`.", "",
             "## The 28 audit items", "",
             "| # | Item | File | Symbol checked | Behaviour in the code | This vs original EfficientNet-B3 | This vs ResNet-50 |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['item']} | `{r['file']}` | `{r['symbol']}` | {r['behaviour']} | {r['this_vs_reference']} | {r['this_vs_resnet50']} |")
    lines += ["", "\"not in both\" means the reference keeps that value elsewhere: the configuration file is new, and "
              "`tests/test_00_reference_integrity.py` asserts its values equal the reference's. Against ResNet-50, \"differs\" on `models/model.py` covers the whole file, although only the backbone, import and classifier width lines differ; on the configuration file, `tests/test_05_fixed_mapping_and_parity.py` confirms only labels, paths and model dimensions differ.", "",
              "## Every source file", "", "| File | vs reference | vs ResNet-50 |", "|---|---|---|"]
    lines += [f"| `{r['file']}` | {r['vs_reference']} | {r['vs_resnet50']} |" for r in table]
    lines += ["", "## Differences between the implementations", "",
              "**Original EfficientNet-B3 vs ResNet-50:** the backbone (D1), and the execution differences D2 to D11 listed below.", "",
              "**Original EfficientNet-B3 vs this experiment:** no model difference. Execution differences only:", "",
              "- D2: Ray simulation replaced by the in-process client proxy; Flower's `Server.fit_round` and `evaluate_round` still drive each round.",
              "- D3: fixed client-to-hospital mapping. This changes the data behind each client, so it can change results.",
              "- D4: pixel-identical image cache.", "- D5: `num_workers` 0.", "- D6: one shared model object across clients.",
              "- D7: torch seeds added.", "- D8: fail-fast on client errors.", "- D9: extra diagnostics such as evaluation focal loss.",
              "- D10: resume state.", "- D11: evaluation wrapper around the unmodified `evaluate.py`.", "",
              "**ResNet-50 vs this experiment:**", "",
              "- The backbone: model file, pretrained weights and dimensions.",
              "- `timm` 1.0.19 and its two dependencies are present here, in `.deps`.",
              "- `release_memory_before_aggregation` is in both code trees, but no round of `resnet50-seed42` used it. Every round of this run will. It frees stale arrays only and does not touch weights or formulas.",
              "- Same seed does not mean same random numbers in training: creating a different model consumes the torch generator differently, so the head initialisation and the data-loader shuffling differ. The participation schedule is unaffected, because it uses its own seeded generator.", ""]
    with open(os.path.join(ROOT, "documentation", "AUDIT_REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("items", len(rows), "| vs reference:", {s: sum(r["this_vs_reference"] == s for r in rows) for s in ("identical", "differs", "not in both")},
          "| vs resnet:", {s: sum(r["this_vs_resnet50"] == s for r in rows) for s in ("identical", "differs", "not in both")})
    print("files differing from ResNet-50:", [r["file"] for r in table if r["vs_resnet50"] != "identical"])
    print("reference files differing:", [r["file"] for r in table if r["vs_reference"] == "differs"])


if __name__ == "__main__":
    main()

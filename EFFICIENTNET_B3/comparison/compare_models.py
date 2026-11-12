"""Compares the two fixed-mapping runs under the rules fixed in chapter 8 of the plan.

Reads each run's saved predictions, logs and checkpoints; nothing is typed in by hand and no model is
run. The ResNet-50 folder is only read. Writes results/comparison/COMPARISON_RESULTS.json and .csv,
plus plots/.

Uncertainty: paired, patient-clustered bootstrap. Each resample draws patients, and both models'
predictions for those patients go in together. A difference is called clear only if its 95% interval
excludes zero.

Usage: python comparison/compare_models.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
RESNET_ROOT = os.path.abspath(os.path.join(ROOT, "..", "RESNET 50_ fed novelty", "RESNET50_EFFICIENTNET_B3_EQUIVALENT"))
REFERENCE_RESULTS = os.path.abspath(os.path.join(ROOT, "..", "..", "FedMelanoma_repo", "final_test_results.txt"))
PARTITIONS = os.path.abspath(os.path.join(ROOT, "..", "without_federated", "client", "hospital_partitions"))
OUT = os.path.join(ROOT, "results", "comparison")
PLOTS = os.path.join(OUT, "plots")
RESAMPLES, SEED = 2000, 42

MODELS = {
    "resnet50": {"label": "ResNet-50", "root": RESNET_ROOT, "run": "resnet50-seed42", "color": "#2a78d6", "marker": "o"},
    "efficientnet_b3": {"label": "EfficientNet-B3", "root": ROOT, "run": "efficientnetb3-fixedmap-seed42", "color": "#eb6834", "marker": "s"},
}
TAGS = ("final_best", "final_round50")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BOOTSTRAP_METRICS = ("roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy")
NAMES = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "recall": "Recall", "precision": "Precision", "f1": "F1", "accuracy": "Accuracy"}


def load_json(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def results_dir(m):
    return os.path.join(m["root"], "results", m["run"])


# ---------------------------------------------------------------- classification
def point_metrics(y, p):
    """ROC-AUC and PR-AUC computed as evaluate.py computes them; class calls at 0.5."""
    pred = p >= 0.5
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    curve_p, curve_r, _ = precision_recall_curve(y, p)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": auc(curve_r, curve_p), "recall": recall, "precision": precision,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "accuracy": float((pred == (y == 1)).mean())}


def paired_bootstrap(frame):
    """frame columns: patient, y_true, p_a (ResNet-50), p_b (EfficientNet-B3). Difference is b minus a."""
    groups = list(frame.groupby("patient").indices.values())
    y, pa, pb = frame.y_true.to_numpy(), frame.p_a.to_numpy(), frame.p_b.to_numpy()
    rng = np.random.default_rng(SEED)
    a, b = {k: [] for k in BOOTSTRAP_METRICS}, {k: [] for k in BOOTSTRAP_METRICS}
    for _ in range(RESAMPLES):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if not 0 < y[rows].sum() < len(rows):
            continue
        ma, mb = point_metrics(y[rows], pa[rows]), point_metrics(y[rows], pb[rows])
        for k in BOOTSTRAP_METRICS:
            a[k].append(ma[k])
            b[k].append(mb[k])
    ma, mb = point_metrics(y, pa), point_metrics(y, pb)
    out = {"rows": len(frame), "malignant": int(y.sum()), "patients": len(groups), "resamples": len(a["roc_auc"]), "metrics": {}}
    for k in BOOTSTRAP_METRICS:
        diff = np.array(b[k]) - np.array(a[k])
        low, high = np.percentile(diff, [2.5, 97.5])
        out["metrics"][k] = {"resnet50": ma[k], "efficientnet_b3": mb[k], "difference": mb[k] - ma[k], "difference_95": [float(low), float(high)],
                             "clear": bool(low > 0 or high < 0), "resnet50_95": [float(x) for x in np.percentile(a[k], [2.5, 97.5])],
                             "efficientnet_b3_95": [float(x) for x in np.percentile(b[k], [2.5, 97.5])]}
    return out


def read_split(split):
    columns = ["isic_id", "patient_id", "lesion_id", "diagnosis_confirm_type"]
    frames = [pd.read_csv(os.path.join(PARTITIONS, h, f"metadata_{split}.csv"), usecols=columns, low_memory=False)
              for h in sorted(os.listdir(PARTITIONS)) if os.path.isdir(os.path.join(PARTITIONS, h))]
    return pd.concat(frames, ignore_index=True)


def classification():
    train, test = read_split("train"), read_split("test")
    test = test.assign(patient=test.patient_id.fillna(test.isic_id), seen_patient=test.patient_id.isin(set(train.patient_id.dropna())),
                       seen_lesion=test.lesion_id.isin(set(train.lesion_id.dropna())),
                       histo=test.diagnosis_confirm_type.fillna("").str.contains("histopathology", case=False))
    out, frames = {}, {}
    for tag in TAGS:
        a = pd.read_csv(os.path.join(results_dir(MODELS["resnet50"]), tag, "test_predictions.csv"))
        b = pd.read_csv(os.path.join(results_dir(MODELS["efficientnet_b3"]), tag, "test_predictions.csv"))
        assert list(a.isic_id) == list(b.isic_id) and list(a.y_true) == list(b.y_true) and len(a) == 1952, "test rows differ"
        frame = a.rename(columns={"y_prob": "p_a"})[["isic_id", "hospital", "y_true", "p_a"]].assign(p_b=b.y_prob.to_numpy())
        frame = frame.merge(test, on="isic_id", how="left", validate="one_to_one")
        assert len(frame) == 1952 and frame.patient.notna().all()
        frames[tag] = frame
        views = {"all_rows": frame, "histopathology_confirmed_only": frame[frame.histo],
                 "neither_patient_nor_lesion_seen": frame[~frame.seen_patient & ~frame.seen_lesion]}
        out[tag] = {name: paired_bootstrap(view) for name, view in views.items()}
        melanoma = frame[frame.y_true == 1]
        ra, rb = melanoma.p_a >= 0.5, melanoma.p_b >= 0.5
        benign = frame[frame.y_true == 0]
        fa, fb = benign.p_a >= 0.5, benign.p_b >= 0.5
        out[tag]["agreement"] = {"melanomas_found_by_both": int((ra & rb).sum()), "melanomas_found_by_resnet50_only": int((ra & ~rb).sum()),
                                 "melanomas_found_by_efficientnet_b3_only": int((~ra & rb).sum()), "melanomas_missed_by_both": int((~ra & ~rb).sum()),
                                 "false_alarms_by_both": int((fa & fb).sum()), "false_alarms_resnet50_only": int((fa & ~fb).sum()),
                                 "false_alarms_efficientnet_b3_only": int((~fa & fb).sum()),
                                 "probability_correlation_spearman": float(frame.p_a.corr(frame.p_b, method="spearman"))}
    return out, frames


def historical():
    text = open(REFERENCE_RESULTS, encoding="utf-8").read()

    def grab(label):
        return float(re.search(rf"{label}\s*:\s*([\d.]+)", text).group(1))

    tn_fp = re.search(r"True Benign\s+([\d,]+)\s+([\d,]+)", text)
    fn_tp = re.search(r"True Malignant\s+([\d,]+)\s+([\d,]+)", text)
    tn, fp, fn, tp = (int(x.replace(",", "")) for x in (*tn_fp.groups(), *fn_tp.groups()))
    return {"accuracy": grab("Accuracy") / 100, "precision": grab("Precision"), "recall": grab("Recall"), "f1": grab("F1 Score"), "roc_auc": grab("ROC-AUC"),
            "pr_auc": grab("PR-AUC"), "tn": tn, "fp": fp, "fn": fn, "tp": tp, "note": "Historical, indicative only: different mapping and transport; no per-image predictions."}


# ---------------------------------------------------------------- architecture, federation, compute
def architecture(m):
    ckpt = os.path.join(m["root"], "checkpoints", m["run"], "global_round_0050.pth")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    buffers = ("running_mean", "running_var", "num_batches_tracked")
    learned = {k: v for k, v in state.items() if not k.endswith(buffers)}
    backbone = sum(v.numel() for k, v in learned.items() if k.startswith("backbone."))
    weight = next(v for k, v in state.items() if k.startswith("classifier") and k.endswith("weight"))
    return {"state_tensors": len(state), "parameters": sum(v.numel() for v in learned.values()), "backbone_parameters": backbone,
            "non_backbone_parameters": sum(v.numel() for v in learned.values()) - backbone, "fusion_inputs": int(weight.shape[1]), "image_features": int(weight.shape[1]) - 32,
            "weights_mb": sum(v.numel() * v.element_size() for v in state.values()) / 1e6, "checkpoint_file_mb": os.path.getsize(ckpt) / 1e6}


def round_table(m):
    return pd.read_csv(os.path.join(results_dir(m), "tables", "per_round_metrics.csv"))


def federation(tables):
    summaries, out = {}, {}
    for key, m in MODELS.items():
        logs = [r for r in load_jsonl(m["root"], "logs", "op", "stragglers", f"{m['run']}.jsonl") if r["event"] == "round_summary"]
        kept = {r["round"]: r for r in logs}  # a crashed round's repeat replaces its earlier record
        summaries[key] = kept
        data = load_json(results_dir(m), "report_data.json")
        part = pd.read_csv(os.path.join(results_dir(m), "tables", "client_participation.csv"))
        out[key] = {**data["stats"], "schedule_matches_plan": data["rounds_compared"] - data["schedule_mismatches"], "hash_failures": data["hash_failures"],
                    "broken_hash_chain": data["broken_hash_chain"], "resumes": data["resumes"], "aggregated_updates_total": int(tables[key].clients_aggregated.sum()),
                    "new_stragglers_total": int(tables[key].new_stragglers.sum()), "failed_total": int(sum(r["n_failed"] for r in kept.values())),
                    "participation": part[["client_id", "hospital", "tier", "selected", "on_time", "late", "failed", "aggregated"]].to_dict("records")}
    fields = ("selected_set", "on_time_set", "straggler_set", "failed_set", "drained_set")
    a, b = summaries["resnet50"], summaries["efficientnet_b3"]
    same = sum(all(a[r][f] == b[r][f] for f in fields) and abs(a[r]["adaptive_window_w_t"] - b[r]["adaptive_window_w_t"]) < 1e-9 for r in range(1, 51))
    out["rounds_with_identical_client_sets_and_window"] = same
    return out


def convergence(tables):
    a, b = tables["resnet50"].set_index("round"), tables["efficientnet_b3"].set_index("round")
    out = {}
    for key, t in (("resnet50", a), ("efficientnet_b3", b)):
        best = int(t.roc_auc.idxmax())
        out[key] = {"best_round": best, "best_selected_clients_roc_auc": float(t.roc_auc.max()),
                    "first_round_at_or_above": {str(x): (int(t.index[t.roc_auc >= x][0]) if (t.roc_auc >= x).any() else None) for x in (0.90, 0.95, 0.97)},
                    "mean_roc_auc": {"rounds_2_10": float(t.roc_auc.loc[2:10].mean()), "rounds_11_30": float(t.roc_auc.loc[11:30].mean()), "rounds_31_50": float(t.roc_auc.loc[31:50].mean())},
                    "std_roc_auc_rounds_31_50": float(t.roc_auc.loc[31:50].std()),
                    "mean_train_loss": {"rounds_2_10": float(t.train_loss.loc[2:10].mean()), "rounds_31_50": float(t.train_loss.loc[31:50].mean())},
                    "mean_eval_focal_loss": {"rounds_2_10": float(t.eval_focal_loss.loc[2:10].mean()), "rounds_11_30": float(t.eval_focal_loss.loc[11:30].mean()), "rounds_31_50": float(t.eval_focal_loss.loc[31:50].mean())}}
    diff = (b.roc_auc - a.roc_auc).loc[2:50]
    out["paired_rounds_2_50"] = {"efficientnet_b3_higher": int((diff > 0).sum()), "resnet50_higher": int((diff < 0).sum()), "mean_difference": float(diff.mean()),
                                 "note": "Both runs evaluate the same selected hospitals in each round. Rounds are descriptive, not independent."}
    return out


def compute(m, table):
    audit = load_jsonl(m["root"], "logs", "audit", f"{m['run']}.jsonl")
    stamps = [e["logged_at"] for e in audit]
    clients = {}
    for r in load_jsonl(m["root"], "logs", "clients", f"{m['run']}.jsonl"):
        if r.get("event") == "client_fit":
            clients[(r["round"], r["client_id"])] = r  # the kept attempt of a repeated round
    fits = pd.DataFrame(clients.values())
    res = pd.read_csv(os.path.join(m["root"], "logs", "resources", f"{m['run']}.csv"))
    data = load_json(results_dir(m), "report_data.json")
    images = int((fits.num_examples * 5).sum())
    return {"wall_seconds": max(stamps) - min(stamps), "kept_round_seconds": float(table.seconds.sum()), "lost_seconds": max(stamps) - min(stamps) - float(table.seconds.sum()),
            "mean_round_seconds": float(table.seconds.mean()), "median_round_seconds": float(table.seconds.median()), "client_fits": len(fits),
            "mean_client_fit_seconds": float(fits.wall_seconds.mean()), "total_client_fit_seconds": float(fits.wall_seconds.sum()),
            "train_images_processed": images, "train_images_per_second": images / float(fits.wall_seconds.sum()),
            "peak_torch_allocated_mb": float(fits.gpu_peak_mb.max()), "peak_device_memory_mb": data["resources"]["gpu_mem_max_mb"], "mean_gpu_util_pct": data["resources"]["gpu_util_mean_pct"],
            "peak_gpu_temp_c": data["resources"]["gpu_temp_max_c"], "mean_gpu_power_w": float(res.gpu_power_w.mean()), "peak_process_rss_gb": data["resources"]["process_rss_max_gb"],
            "min_free_ram_gb": data["resources"]["ram_free_min_gb"], "min_free_commit_gb": data["resources"]["commit_free_min_gb"], "commit_limit_max_gb": data["resources"]["commit_limit_max_gb"],
            "crashes": data["resumes"], "resume_state_mb": os.path.getsize(os.path.join(m["root"], "checkpoints", m["run"], "resume_state.pkl")) / 1e6}


# ---------------------------------------------------------------- plots
def style(ax, title=None, xlabel=None, ylabel=None, grid="y"):
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8.5)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(os.path.join(PLOTS, name), dpi=170)
    plt.close(fig)


def legend(ax, **kw):
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, **kw)


def legend_above(ax):
    """Sits above the plot area, so it can never cover a data point."""
    legend(ax, loc="lower right", bbox_to_anchor=(1, 1), ncol=2, borderaxespad=0, columnspacing=1.2, handlelength=1.8)


def rounds_line(ax, tables, column, first=2):
    for key, m in MODELS.items():
        t = tables[key][tables[key]["round"] >= first]
        ax.plot(t["round"], t[column], color=m["color"], linewidth=1.8, marker=m["marker"], markersize=3, label=m["label"])


def plot_roc(frames, summary):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for ax, tag, name in zip(axes, TAGS, ("Best checkpoints", "Round 50")):
        frame = frames[tag]
        for key, col in (("resnet50", "p_a"), ("efficientnet_b3", "p_b")):
            m = MODELS[key]
            fpr, tpr, _ = roc_curve(frame.y_true, frame[col])
            value = summary[tag]["all_rows"]["metrics"]["roc_auc"][key]
            ax.plot(fpr, tpr, color=m["color"], linewidth=2, linestyle="-" if key == "resnet50" else "--", label=f"{m['label']}: {value:.4f}")
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
        style(ax, name, "False positive rate", "True positive rate" if tag == "final_best" else None, grid="both")
        legend(ax, loc="lower right", title=None)
    save(fig, "roc_comparison.png")


def plot_training(tables):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    rounds_line(axes[0], tables, "train_loss", 1)
    style(axes[0], "Training loss", "Federated round", "Class-weighted focal loss")
    rounds_line(axes[1], tables, "eval_focal_loss", 1)
    style(axes[1], "Evaluation focal loss", "Federated round")
    legend_above(axes[0])
    save(fig, "training_loss_comparison.png")


def plot_auc_rounds(tables, conv):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    rounds_line(ax, tables, "roc_auc")
    for key, m in MODELS.items():
        r = conv[key]["best_round"]
        ax.axvline(r, color=m["color"], linewidth=1, linestyle=":")
        ax.annotate(f"{m['label']} best, round {r}", (r, 0.04 if key == "resnet50" else 0.12), xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK)
    style(ax, "ROC-AUC, rounds 2 to 50", "Federated round", "ROC-AUC, selected clients")
    legend_above(ax)
    save(fig, "roc_auc_across_rounds.png")


def plot_metric(tables, summary, metric, name):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), gridspec_kw={"width_ratios": [2.1, 1]})
    rounds_line(axes[0], tables, metric)
    style(axes[0], f"{NAMES[metric]}, rounds 2 to 50", "Federated round", f"{NAMES[metric]}, selected clients")
    legend_above(axes[0])
    ax = axes[1]
    for i, tag in enumerate(TAGS):
        cell = summary[tag]["all_rows"]["metrics"][metric]
        for j, key in enumerate(MODELS):
            m, y = MODELS[key], i + (0.18 if j == 0 else -0.18)
            low, high = cell[f"{key}_95"]
            ax.plot([low, high], [y, y], color=m["color"], linewidth=2)
            ax.plot(cell[key], y, marker=m["marker"], color=m["color"], markersize=7, markeredgecolor="white")
    ax.set_yticks([0, 1], ["Best", "Round 50"])
    ax.invert_yaxis()
    style(ax, "All 1,952 test images", f"{NAMES[metric]}, with 95% interval", grid="x")
    save(fig, name)


def plot_confusion(paths):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6))
    for row, tag in enumerate(TAGS):
        for col, key in enumerate(MODELS):
            m, ax = MODELS[key], axes[row][col]
            d = load_json(results_dir(m), "report_data.json")
            metrics = d["finals"][tag]
            cells = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=float)
            share = cells / cells.sum(axis=1, keepdims=True)
            ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{int(cells[i, j]):,}\n{share[i, j] * 100:.1f}%", ha="center", va="center", fontsize=11, color="white" if share[i, j] > 0.55 else INK)
            ax.set_xticks([0, 1], ["Pred. benign", "Pred. malignant"])
            ax.set_yticks([0, 1], ["True benign", "True malignant"])
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            for side in ax.spines.values():
                side.set_visible(False)
            when = f"round {d['best']['round']} (best)" if tag == "final_best" else "round 50"
            ax.set_title(f"{m['label']}, {when}", loc="left", fontsize=10, color=INK)
    save(fig, "confusion_matrices_side_by_side.png")


def plot_thresholds(thresholds):
    fig, axes = plt.subplots(2, 3, figsize=(8, 5), sharex=True, sharey=True)
    for row, tag in enumerate(TAGS):
        for col, metric in enumerate(("precision", "recall", "f1")):
            ax = axes[row][col]
            for key, m in MODELS.items():
                ts = thresholds[key][tag]
                ax.plot([t["threshold"] for t in ts], [t[metric] for t in ts], color=m["color"], linewidth=2, marker=m["marker"], markersize=5, label=m["label"])
            style(ax, f"{NAMES[metric]}: {'best' if tag == 'final_best' else 'round 50'}", "Threshold" if row else None, None)
            ax.set_ylim(0.3, 1)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=8, labelcolor=INK)
    save(fig, "threshold_wise_comparison.png", rect=(0, 0, 1, 0.95))


def plot_time(comp, tables):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), gridspec_kw={"width_ratios": [1, 1.7]})
    ax = axes[0]
    for i, key in enumerate(MODELS):
        m, c = MODELS[key], comp[key]
        kept, lost = c["kept_round_seconds"] / 3600, c["lost_seconds"] / 3600
        ax.bar(i, kept, width=0.5, color=m["color"])
        if lost > 0.02:
            ax.bar(i, lost, bottom=kept + 0.03, width=0.5, color=m["color"], alpha=0.35)
        ax.text(i, kept + lost + 0.12, f"{c['wall_seconds'] / 3600:.2f} h", ha="center", fontsize=9, color=INK)
    ax.set_xticks([0, 1], [m["label"] for m in MODELS.values()])
    ax.set_ylim(0, max(c["wall_seconds"] for c in comp.values()) / 3600 * 1.15)
    style(ax, "Total time", None, "Hours (light: lost to crashes)")
    rounds_line(axes[1], tables, "seconds", 1)
    style(axes[1], "Time per round", "Federated round", "Seconds")
    legend_above(axes[1])
    save(fig, "training_time_comparison.png")


def plot_resources(comp, arch):
    items = [("Peak GPU memory, device (MB)", "peak_device_memory_mb", comp), ("Peak allocated by PyTorch (MB)", "peak_torch_allocated_mb", comp), ("Mean GPU utilisation (%)", "mean_gpu_util_pct", comp),
             ("Peak process memory (GB)", "peak_process_rss_gb", comp), ("Lowest free commit (GB)", "min_free_commit_gb", comp), ("Checkpoint size (MB)", "checkpoint_file_mb", arch)]
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.4))
    for ax, (title, field, source) in zip(axes.flat, items):
        values = [source[k][field] for k in MODELS]
        for i, key in enumerate(MODELS):
            ax.bar(i, values[i], width=0.5, color=MODELS[key]["color"])
            ax.text(i, values[i] * 1.03, f"{values[i]:,.2f}" if values[i] < 10 else f"{values[i]:,.0f}", ha="center", fontsize=8.5, color=INK)
        ax.set_xticks([0, 1], ["ResNet-50", "EffNet-B3"])
        ax.set_ylim(0, max(values) * 1.2)
        style(ax, None, None, None)
        ax.set_title(title, loc="left", fontsize=8.8, color=INK)
    save(fig, "resource_utilisation_comparison.png")


def plot_differences(summary):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
    order = list(BOOTSTRAP_METRICS)
    for ax, tag, name in zip(axes, TAGS, ("Best checkpoints", "Round 50")):
        for i, metric in enumerate(order):
            cell = summary[tag]["all_rows"]["metrics"][metric]
            low, high = cell["difference_95"]
            ax.plot([low, high], [i, i], color=MUTED, linewidth=2)
            ax.plot(cell["difference"], i, "o", color=INK if cell["clear"] else "white", markeredgecolor=INK, markersize=6)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_yticks(range(len(order)), [NAMES[k] for k in order])
        ax.invert_yaxis() if tag == "final_best" else None
        style(ax, name, "EfficientNet-B3 minus ResNet-50", grid="x")
    save(fig, "paired_differences.png")


# ---------------------------------------------------------------- output
def write_csv(result):
    rows = []

    def add(section, item, a, b, diff="", low="", high="", clear="", note=""):
        rows.append({"section": section, "item": item, "resnet50": a, "efficientnet_b3": b, "difference_b_minus_a": diff, "difference_95_low": low, "difference_95_high": high, "clear": clear, "note": note})

    for key in result["architecture"]["resnet50"]:
        add("A architecture", key, result["architecture"]["resnet50"][key], result["architecture"]["efficientnet_b3"][key])
    for tag in TAGS:
        for view in ("all_rows", "histopathology_confirmed_only", "neither_patient_nor_lesion_seen"):
            block = result["classification"][tag][view]
            for metric, c in block["metrics"].items():
                add("B classification", f"{tag}/{view}/{metric}", c["resnet50"], c["efficientnet_b3"], c["difference"], c["difference_95"][0], c["difference_95"][1], c["clear"], f"rows {block['rows']}, patients {block['patients']}")
        for name in ("tn", "fp", "fn", "tp"):
            add("B classification", f"{tag}/{name}", result["finals"]["resnet50"][tag][name], result["finals"]["efficientnet_b3"][tag][name])
        for ta, tb in zip(result["thresholds"]["resnet50"][tag], result["thresholds"]["efficientnet_b3"][tag]):
            for metric in ("precision", "recall", "f1", "specificity"):
                add("B thresholds", f"{tag}/t{ta['threshold']:.1f}/{metric}", ta[metric], tb[metric], tb[metric] - ta[metric])
    for field in ("completed_rounds", "empty_rounds", "quorum_label_rounds", "mean_selected", "mean_aggregated", "mean_failed", "mean_new_stragglers", "aggregated_updates_total", "new_stragglers_total", "failed_total", "schedule_matches_plan", "hash_failures", "broken_hash_chain", "resumes"):
        add("C federated", field, result["federation"]["resnet50"][field], result["federation"]["efficientnet_b3"][field])
    for field in ("best_round", "best_selected_clients_roc_auc", "std_roc_auc_rounds_31_50"):
        add("C convergence", field, result["convergence"]["resnet50"][field], result["convergence"]["efficientnet_b3"][field])
    for field in result["compute"]["resnet50"]:
        add("D compute", field, result["compute"]["resnet50"][field], result["compute"]["efficientnet_b3"][field])
    with open(os.path.join(OUT, "COMPARISON_RESULTS.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    os.makedirs(PLOTS, exist_ok=True)
    tables = {k: round_table(m) for k, m in MODELS.items()}
    summary, frames = classification()
    reports = {k: load_json(results_dir(m), "report_data.json") for k, m in MODELS.items()}
    conv = convergence(tables)
    arch = {k: architecture(m) for k, m in MODELS.items()}
    comp = {k: compute(m, tables[k]) for k, m in MODELS.items()}
    thresholds = {k: {tag: reports[k]["finals"][tag]["thresholds"] for tag in TAGS} for k in MODELS}
    result = {"rules": "Chapter 8 of EFFICIENTNET_B3_FIXED_MAPPING_PLAN.pdf", "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "difference": "efficientnet_b3 minus resnet50"},
              "architecture": arch, "classification": summary,
              "finals": {k: {tag: {x: y for x, y in reports[k]["finals"][tag].items() if x not in ("thresholds", "checkpoint")} for tag in TAGS} for k in MODELS},
              "thresholds": thresholds, "convergence": conv, "federation": federation(tables), "compute": comp, "historical_efficientnet_b3": historical()}
    with open(os.path.join(OUT, "COMPARISON_RESULTS.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    write_csv(result)
    plot_roc(frames, summary)
    plot_training(tables)
    plot_auc_rounds(tables, conv)
    plot_metric(tables, summary, "accuracy", "accuracy_comparison.png")
    plot_metric(tables, summary, "f1", "f1_comparison.png")
    plot_confusion(None)
    plot_thresholds(thresholds)
    plot_time(comp, tables)
    plot_resources(comp, arch)
    plot_differences(summary)
    print(json.dumps({k: v for k, v in result.items() if k not in ("thresholds", "finals")}, indent=0, default=str)[:200])
    for tag in TAGS:
        for view in ("all_rows", "histopathology_confirmed_only", "neither_patient_nor_lesion_seen"):
            b = summary[tag][view]
            print(tag, view, b["rows"], b["patients"], b["resamples"])
            for k, c in b["metrics"].items():
                print(f"   {k:10s} R {c['resnet50']:.4f} E {c['efficientnet_b3']:.4f} d {c['difference']:+.4f} [{c['difference_95'][0]:+.4f},{c['difference_95'][1]:+.4f}] clear={c['clear']}")
        print(summary[tag]["agreement"])
    print(json.dumps(conv, indent=0))
    print(json.dumps(arch, indent=0))
    print(json.dumps(comp, indent=0))
    print({k: v for k, v in result["federation"]["efficientnet_b3"].items() if k != "participation"})
    print({k: v for k, v in result["federation"]["resnet50"].items() if k != "participation"})
    print("identical rounds", result["federation"]["rounds_with_identical_client_sets_and_window"])
    print(result["historical_efficientnet_b3"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

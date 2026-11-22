"""Novelty hyperparameter tuning setup (2026-11-21). Creates only new files under /workspace/novelty_hpo.

1. Validation split: 15% of TRAINING patients (global patient split, so a patient never sits in both; stratified by
   whether the patient has a malignant image), seed 42. Per hospital: data_val/<H>/metadata_train.csv (remaining
   rows) and data_val/<H>/metadata_test.csv (= the validation rows; the code evaluates its "test" split each round).
   images/Train and images/Test are symlinks to the ORIGINAL images/Train (validation images are training images).
   The real test set is never read during tuning.
2. Cache: rows sliced out of /workspace/six_models_dgmm/cache_dgmm/<H>__train.npy (no image re-decoding).
3. Trials: H00 = current defaults + 9 random draws from the grid (seed 2026) -> code/ablation/hpo_trials.json and
   registry entries in code/ablation/experiments.yaml.
4. Code hook (copy only): common.build_run_config sets learning_rate and cfg["hpo"]; run_one.server_fn sets the
   selector's tau, gamma, lambda_, rho after apply_variant and logs an audit record.
"""
import json, os, hashlib, sys
import numpy as np, pandas as pd

H = "/workspace/novelty_hpo"
CODE = H + "/code"
ORIG = "/workspace/hospital_two_feature/fed_ready/3_Dirichlet_GMM"
OCACHE = "/workspace/six_models_dgmm/cache_dgmm"
VAL, VCACHE = H + "/data_val", H + "/cache_val"
sys.path.insert(0, CODE)
from client.dataset import MelanomaDataset  # noqa: E402

hosp = sorted(d for d in os.listdir(ORIG) if os.path.isdir(os.path.join(ORIG, d)))
tables, labels = {}, {}
for h in hosp:
    csv = os.path.join(ORIG, h, "metadata_train.csv")
    tables[h] = pd.read_csv(csv, dtype=str, keep_default_na=False)
    ds = MelanomaDataset(csv, os.path.join(ORIG, h, "images", "Train"))
    lab = dict(zip([r["isic_id"] for r in ds.rows], [int(v) for v in ds.labels]))
    tables[h]["_y"] = tables[h].isic_id.map(lab)
allp = pd.concat([t[["patient_id", "_y"]] for t in tables.values()])
assert allp._y.notna().all()
pat = allp.groupby("patient_id")._y.max()
rng = np.random.default_rng(42)
val_pat = set()
for cls in (0, 1):
    ps = np.array(sorted(pat[pat == cls].index))
    val_pat |= set(rng.choice(ps, size=int(round(0.15 * len(ps))), replace=False))
print("train patients", len(pat), "validation patients", len(val_pat))

os.makedirs(VCACHE, exist_ok=True)
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
tot = {"train": [0, 0], "val": [0, 0]}
for h in hosp:
    t = tables[h]; isval = t.patient_id.isin(val_pat)
    d = os.path.join(VAL, h); os.makedirs(os.path.join(d, "images"), exist_ok=True)
    for sub in ("Train", "Test"):
        link = os.path.join(d, "images", sub)
        if not os.path.islink(link): os.symlink(os.path.join(ORIG, h, "images", "Train"), link)
    cols = [c for c in t.columns if c != "_y"]
    t.loc[~isval, cols].to_csv(os.path.join(d, "metadata_train.csv"), index=False)
    t.loc[isval, cols].to_csv(os.path.join(d, "metadata_test.csv"), index=False)
    old = np.load(os.path.join(OCACHE, h + "__train.npy"), mmap_mode="r")
    old_ids = json.load(open(os.path.join(OCACHE, h + "__train.json")))["isic_ids"]
    pos = {i: k for k, i in enumerate(old_ids)}
    for split, csvname in (("train", "metadata_train.csv"), ("test", "metadata_test.csv")):
        csv = os.path.join(d, csvname)
        ds = MelanomaDataset(csv, os.path.join(d, "images", "Train"))
        ids = [r["isic_id"] for r in ds.rows]
        arr = np.ascontiguousarray(old[[pos[i] for i in ids]])
        np.save(os.path.join(VCACHE, h + "__" + split + ".npy"), arr)
        json.dump({"csv_sha256": sha(csv), "isic_ids": ids}, open(os.path.join(VCACHE, h + "__" + split + ".json"), "w"))
        y = np.array([int(v) for v in ds.labels]); key = "train" if split == "train" else "val"
        tot[key][0] += len(y); tot[key][1] += int(y.sum())
        print(h, "train" if split == "train" else "val  ", "n", len(y), "malignant", int(y.sum()))
print("TOTAL", tot)

GRID = {"learning_rate": [5e-5, 1e-4, 2e-4, 3e-4], "tau": [0.1, 0.2, 0.35, 0.5], "gamma": [0.5, 1.0, 1.5, 2.0],
        "lambda": [0.0, 0.02, 0.05, 0.1], "rho": [0.7, 0.85, 0.95], "focal_gamma": [1.0, 2.0, 3.0]}
trials = {"H00": {"learning_rate": 1e-4, "tau": 0.2, "gamma": 1.0, "lambda": 0.05, "rho": 0.85, "focal_gamma": 2.0}}
r2 = np.random.default_rng(2026); seen = {json.dumps(trials["H00"], sort_keys=True)}
while len(trials) < 10:
    c = {k: float(v[r2.integers(len(v))]) for k, v in GRID.items()}
    s = json.dumps(c, sort_keys=True)
    if s in seen: continue
    seen.add(s); trials["H%02d" % len(trials)] = c
json.dump(trials, open(CODE + "/ablation/hpo_trials.json", "w"), indent=1)
for k, v in trials.items(): print(k, v)

reg = CODE + "/ablation/experiments.yaml"; text = open(reg).read()
if "  H00:" not in text:
    add = "".join("  %s: {focal_gamma: %s}\n" % (k, v["focal_gamma"]) for k, v in trials.items())
    assert text.count("experiments:\n") == 1
    text = text.replace("experiments:\n", "experiments:\n" + add); open(reg, "w").write(text)

cm = CODE + "/ablation/common.py"; t = open(cm).read()
if "hpo_trials.json" not in t:
    old = '    cfg["local_training"]["focal_gamma"] = float(variant["focal_gamma"])\n'
    assert t.count(old) == 1
    new = old + ('    _hpo_path = os.path.join(ROOT, "ablation", "hpo_trials.json")\n'
                 '    _hpo = json.load(open(_hpo_path)).get(exp_id) if os.path.exists(_hpo_path) else None\n'
                 '    if _hpo:   # hyperparameter tuning (2026-11-21): learning rate here, selector constants in run_one\n'
                 '        cfg["local_training"]["learning_rate"] = float(_hpo["learning_rate"])\n'
                 '        assert float(_hpo["focal_gamma"]) == cfg["local_training"]["focal_gamma"]\n'
                 '        cfg["hpo"] = dict(_hpo)\n')
    t = t.replace(old, new)
    if "\nimport json" not in t: t = "import json\n" + t
    open(cm, "w").write(t)
ro = CODE + "/ablation/run_one.py"; t = open(ro).read()
if "hpo_override" not in t:
    old = '        apply_variant(components.strategy, ablation["variant"], ablation["harness"], args.seed)\n'
    assert t.count(old) == 1
    new = old + ('        hpo = cfg.get("hpo")\n'
                 '        if hpo:\n'
                 '            s = components.strategy._selector\n'
                 '            s.tau, s.gamma, s.lambda_, s.rho = (float(hpo["tau"]), float(hpo["gamma"]),\n'
                 '                                                float(hpo["lambda"]), float(hpo["rho"]))\n'
                 '            components.strategy._logger.log_audit({"event": "hpo_override", **hpo,\n'
                 '                "selector": [s.tau, s.gamma, s.lambda_, s.rho], "selector_class": type(s).__name__})\n')
    open(ro, "w").write(t.replace(old, new))
print("SETUP DONE")

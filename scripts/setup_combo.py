"""Combined follow-up (user 2026-11-22): C01 = default H00 + aggregation gamma 0.5 + MAD k 2.0 (the two settings that each
reproduced the recall/fairness gain in F02/F04). C00 = an exact copy of H00, run at the same time as a fresh concurrent control."""
import json, shutil
C = "/workspace/novelty_hpo/code/ablation/"
shutil.copy(C + "hpo_trials.json", C + "hpo_trials.json.bak_v4"); shutil.copy(C + "experiments.yaml", C + "experiments.yaml.bak_v4")
t = json.load(open(C + "hpo_trials.json"))
assert "C00" not in t and "C01" not in t
t["C00"] = dict(t["H00"])
d = dict(t["H00"]); assert d["agg_gamma"] != 0.5 and d["mad_k"] != 2.0; d["agg_gamma"] = 0.5; d["mad_k"] = 2.0; t["C01"] = d
json.dump(t, open(C + "hpo_trials.json", "w"), indent=1)
y = open(C + "experiments.yaml").read()
if "  C00:" not in y:
    y = y.replace("experiments:\n", "experiments:\n" + "".join(f"  {f}: {{focal_gamma: {t[f]['focal_gamma']}}}\n" for f in ("C00", "C01")), 1)
    open(C + "experiments.yaml", "w").write(y)
for f in ("C00", "C01"): print(f, {k: v for k, v in t[f].items() if t["H00"][k] != v})

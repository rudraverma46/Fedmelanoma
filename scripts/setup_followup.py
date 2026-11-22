"""Follow-up (user 2026-11-22): one-factor-at-a-time test of the 4 settings shared by H10 and H09, each applied alone to the
default H00. F01 selection gamma 2.0, F02 aggregation gamma 0.5, F03 lambda 0.02, F04 MAD k 2.0. Same protocol as stage 2."""
import json, shutil
C = "/workspace/novelty_hpo/code/ablation/"
shutil.copy(C + "hpo_trials.json", C + "hpo_trials.json.bak_v3"); shutil.copy(C + "experiments.yaml", C + "experiments.yaml.bak_v3")
t = json.load(open(C + "hpo_trials.json"))
CH = {"F01": ("gamma", 2.0), "F02": ("agg_gamma", 0.5), "F03": ("lambda", 0.02), "F04": ("mad_k", 2.0)}
for f, (k, v) in CH.items():
    d = dict(t["H00"]); assert d[k] != v, (k, d[k]); d[k] = v; t[f] = d
json.dump(t, open(C + "hpo_trials.json", "w"), indent=1)
y = open(C + "experiments.yaml").read()
if "  F01:" not in y:
    y = y.replace("experiments:\n", "experiments:\n" + "".join(f"  {f}: {{focal_gamma: {t[f]['focal_gamma']}}}\n" for f in CH), 1)
    open(C + "experiments.yaml", "w").write(y)
for f in CH: print(f, {k: v for k, v in t[f].items() if t["H00"][k] != v})

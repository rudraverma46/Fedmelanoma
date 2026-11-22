"""Stage-1 selection over seeds 42, 43, 44: per trial and seed, mean VALIDATION ROC-AUC over the last 3 rounds;
rank trials by the mean of that over the 3 seeds (ties -> mean final-round AUC). Every trial must have all 3 seeds.
Writes results/stage1_table.csv (one row per trial), results/stage1_per_seed.csv, logs/selection.json."""
import json, os
import pandas as pd
H = "/workspace/novelty_hpo"; SEEDS = (42, 43, 44)
trials = json.load(open(H + "/code/ablation/hpo_trials.json"))
per, rows = [], []
for t, hp in trials.items():
    vals = []
    for s in SEEDS:
        d = f"{H}/raw_stage1/{t}/seed_{s}"
        if not os.path.exists(d + "/DONE.json"):
            per.append({"trial": t, "seed": s, "status": "NOT DONE"}); continue
        m = pd.read_csv(d + "/metrics.csv")
        r = {"trial": t, "seed": s, "status": "DONE", "rounds": int(m["round"].max()),
             "val_auc_last3": float(m.eval_roc_auc.tail(3).mean()), "val_auc_final": float(m.eval_roc_auc.iloc[-1]),
             "val_auc_best": float(m.eval_roc_auc.max()), "val_pr_auc_last3": float(m.eval_pr_auc.tail(3).mean())}
        per.append(r); vals.append(r)
    v = pd.DataFrame(vals)
    row = {"trial": t, **hp, "seeds_done": len(vals)}
    if len(vals):
        row.update({"val_auc_last3_mean": v.val_auc_last3.mean(), "val_auc_last3_sd": v.val_auc_last3.std(ddof=1) if len(v) > 1 else float("nan"),
                    "val_auc_last3_min": v.val_auc_last3.min(), "val_auc_final_mean": v.val_auc_final.mean(),
                    "val_pr_auc_last3_mean": v.val_pr_auc_last3.mean()})
        for _, r in v.iterrows(): row[f"val_auc_last3_s{int(r.seed)}"] = r.val_auc_last3
    rows.append(row)
os.makedirs(H + "/results", exist_ok=True)
pd.DataFrame(per).to_csv(H + "/results/stage1_per_seed.csv", index=False)
tab = pd.DataFrame(rows)
ok = tab[tab.seeds_done == len(SEEDS)].sort_values(["val_auc_last3_mean", "val_auc_final_mean"], ascending=False)
tab.sort_values("val_auc_last3_mean", ascending=False).to_csv(H + "/results/stage1_table.csv", index=False)
assert len(ok) == len(trials), f"only {len(ok)} of {len(trials)} trials have all seeds"
top = [t for t in ok.trial if t != "H00"][:2]
d0 = ok.set_index("trial").loc["H00", "val_auc_last3_mean"]
json.dump({"seeds": list(SEEDS), "top2_non_default": top, "ranking": list(ok.trial), "default_val_auc_last3_mean": float(d0),
           "default_rank": int(list(ok.trial).index("H00")) + 1}, open(H + "/logs/selection.json", "w"), indent=1)
print(ok.round(4).to_string(index=False)); print("SELECTED", top)

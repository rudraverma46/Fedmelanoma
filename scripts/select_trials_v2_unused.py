"""Stage-1 selection: rank trials by mean VALIDATION ROC-AUC over the last 3 rounds (13-15); ties -> best-round AUC.
Writes logs/selection.json (top 2 ids) and results/stage1_table.csv. Uses only validation data."""
import json, os
import pandas as pd
H = "/workspace/novelty_hpo"
trials = json.load(open(H + "/code/ablation/hpo_trials.json"))
rows = []
for t, hp in trials.items():
    d = f"{H}/raw_stage1/{t}/seed_42"
    if not os.path.exists(d + "/DONE.json"):
        rows.append({"trial": t, **hp, "status": "NOT DONE"}); continue
    m = pd.read_csv(d + "/metrics.csv")
    rows.append({"trial": t, **hp, "status": "DONE", "rounds": int(m["round"].max()),
                 "val_auc_last3": float(m.eval_roc_auc.tail(3).mean()), "val_auc_final": float(m.eval_roc_auc.iloc[-1]),
                 "val_auc_best": float(m.eval_roc_auc.max()), "best_round": int(m.loc[m.eval_roc_auc.idxmax(), "round"]),
                 "val_pr_auc_last3": float(m.eval_pr_auc.tail(3).mean()), "val_bal_acc_last3": float(m.eval_balanced_accuracy.tail(3).mean()),
                 "val_recall_last3": float(m.eval_recall.tail(3).mean()), "val_specificity_last3": float(m.eval_specificity.tail(3).mean()),
                 "val_mcc_last3": float(m.eval_mcc.tail(3).mean()), "sec_per_round": float(m.round_seconds.mean())})
df = pd.DataFrame(rows)
ok = df[df.status == "DONE"].sort_values(["val_auc_last3", "val_auc_best"], ascending=False)
os.makedirs(H + "/results", exist_ok=True)
pd.concat([ok, df[df.status != "DONE"]]).to_csv(H + "/results/stage1_table.csv", index=False)
top = [t for t in ok.trial if t != "H00"][:2]
json.dump({"top2_non_default": top, "ranking": list(ok.trial)}, open(H + "/logs/selection.json", "w"), indent=1)
print(ok.round(4).to_string(index=False)); print("SELECTED", top)

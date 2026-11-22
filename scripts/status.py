import glob, os, time, json
import pandas as pd
H = "/workspace/novelty_hpo"
for stage in ("raw_stage1", "raw_stage2"):
    for d in sorted(glob.glob(f"{H}/{stage}/*/seed_42")):
        t = os.path.basename(os.path.dirname(d)); m = d + "/metrics.csv"
        done = os.path.exists(d + "/DONE.json")
        if os.path.exists(m):
            x = pd.read_csv(m)
            print(stage, t, "rounds", int(x["round"].max()), "auc_last", round(float(x.eval_roc_auc.iloc[-1]), 4),
                  "auc_last3", round(float(x.eval_roc_auc.tail(3).mean()), 4), "sec/round", round(float(x.round_seconds.mean())), "DONE" if done else "")
        else:
            st = json.load(open(d + "/status.json")) if os.path.exists(d + "/status.json") else {}
            print(stage, t, "no metrics yet", str(st)[:150])
os.system("df -h /workspace | tail -1; tail -2 /workspace/novelty_hpo/logs/stage1.log; ls /workspace/novelty_hpo/code/checkpoints 2>/dev/null | wc -l")

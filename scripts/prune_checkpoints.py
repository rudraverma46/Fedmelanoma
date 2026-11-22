"""Keeps disk use bounded (21 GB free): in every checkpoints folder of the six runs, delete per-round
global_round_*.pth files except the two newest. best_global_model.pth, final_global_model.pth, resume state and
FedStrag snapshots are never touched. Stops when all six runs have a DONE or STOPPED status."""
import glob, os, time
B = "/workspace/six_models_dgmm"
NAMES = ["fedavg", "fedprox", "novelty", "fedlesscan", "fedasync", "fedstrag"]
def finished(n):
    p = os.path.join(B, "logs", n + ".status")
    if not os.path.exists(p): return False
    t = open(p).read()
    return "DONE" in t or "STOPPED" in t or "EVAL FAILED final" in t
while True:
    for d in sorted({os.path.dirname(p) for p in glob.glob(B + "/src_*/**/checkpoints/**/global_round_*.pth", recursive=True)}):
        files = sorted(glob.glob(os.path.join(d, "global_round_*.pth")))
        for f in files[:-2]:
            try:
                os.remove(f); print(time.strftime("%F %T"), "removed", f, flush=True)
            except FileNotFoundError:
                pass
    if all(finished(n) for n in NAMES):
        print("all finished", flush=True); break
    time.sleep(60)

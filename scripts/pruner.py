import glob, os, shutil, time
C = "/workspace/novelty_hpo/code/checkpoints"
RAWS = ["/workspace/novelty_hpo/raw_stage1", "/workspace/novelty_hpo/raw_stage2"]
LOG = "/workspace/novelty_hpo/logs/pruner.log"
def say(m): open(LOG, "a").write(time.strftime("%F %T ") + m + "\n")
say("pruner start: keep the latest 2 global_round_*.pth per run; remove a run's folder once its DONE.json exists")
while not os.path.exists("/workspace/novelty_hpo/logs/STOP_PRUNER"):
    for d in sorted(glob.glob(C + "/*")):
        exp = os.path.basename(d).split("-s42-")[0]
        if any(os.path.exists(f"{r}/{exp}/seed_42/DONE.json") for r in RAWS):
            shutil.rmtree(d, ignore_errors=True); say("done, removed " + d); continue
        fs = sorted(glob.glob(d + "/global_round_*.pth"), key=lambda f: int(''.join(ch for ch in os.path.basename(f) if ch.isdigit()) or 0))
        for f in fs[:-2]:
            try: os.remove(f)
            except FileNotFoundError: pass
    time.sleep(60)
say("pruner stop")

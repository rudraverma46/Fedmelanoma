import glob, json, os, shutil, time
# v3: a checkpoint folder is removed only when the DONE.json of the SAME run_id exists (any seed, any stage).
# v2 matched on "<trial>-s42-" only, so stage-2 runs (same trial names) would have lost their checkpoints mid-run.
C = "/workspace/novelty_hpo/code/checkpoints"
RAWS = ["/workspace/novelty_hpo/raw_stage1", "/workspace/novelty_hpo/raw_stage2", "/workspace/novelty_hpo/raw_followup"]
LOG = "/workspace/novelty_hpo/logs/pruner.log"
def say(m): open(LOG, "a").write(time.strftime("%F %T ") + m + "\n")
def done_ids():
    ids = set()
    for r in RAWS:
        for f in glob.glob(r + "/*/seed_*/DONE.json"):
            try: ids.add(json.load(open(f))["run_id"])
            except Exception: pass
    return ids
say("pruner v4 start: keep latest 2 global_round_*.pth per run; remove a run folder once DONE.json with the same run_id exists")
while not os.path.exists("/workspace/novelty_hpo/logs/STOP_PRUNER"):
    ids = done_ids()
    for d in sorted(glob.glob(C + "/*")):
        if os.path.basename(d) in ids:
            shutil.rmtree(d, ignore_errors=True); say("done, removed " + d); continue
        fs = sorted(glob.glob(d + "/global_round_*.pth"), key=lambda f: int(''.join(ch for ch in os.path.basename(f) if ch.isdigit()) or 0))
        for f in fs[:-2]:
            try: os.remove(f)
            except FileNotFoundError: pass
    time.sleep(60)
say("pruner v4 stop")

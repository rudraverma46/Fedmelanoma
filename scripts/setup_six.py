"""Setup of the six models on Dirichlet-GMM (2026-11-21). Edits copies under /workspace/six_models_dgmm only."""
import os, re, sys
BASE = "/workspace/six_models_dgmm"
DATA = "/workspace/hospital_two_feature/fed_ready/3_Dirichlet_GMM"
CACHE = BASE + "/cache_dgmm"
MODELS = {
    "fedasync": "src_resnet50-fedasync/resnet50_fedasync",
    "fedavg": "src_resnet50-fedavg-textbook/resnet50_fedavg_textbook",
    "fedprox": "src_resnet50-fedprox-textbook/resnet50_fedprox_textbook",
    "novelty": "src_resnet50-no-confirm-type/RESNET50_NOVELTY MODEL",
    "fedlesscan": "src_resnet_50_fedlesscan/resnet50_fedlesscan",
    "fedstrag": "src_resnet_50_fedstrag",
}
def sub_line(text, key, value, path):
    pat = re.compile(r"^(\s+" + key + r":)[^\n]*$", re.M)
    new, n = pat.subn(lambda m: m.group(1) + " " + value, text, count=1)
    if n != 1: sys.exit(f"{path}: key {key} not found")
    return new
for name, rel in MODELS.items():
    d = os.path.join(BASE, rel)
    c = os.path.join(d, "configs", "experiment_config.yaml")
    t = open(c, encoding="utf-8").read()
    t = sub_line(t, "partitions_root", DATA + "   # Dirichlet-GMM, diagnosis_confirm_type removed (friend's fed_ready, read-only)", c)
    t = sub_line(t, "cache_dir", CACHE + "   # shared image cache built 2026-11-21 (read-only during training)", c)
    t = sub_line(t, "run_id", f"{name}-dgmm-seed42", c)
    open(c, "w", encoding="utf-8").write(t)
    e = os.path.join(d, "evaluation", "evaluate_checkpoint.py")
    s = open(e, encoding="utf-8").read()
    marker = "    cfg = load_config()\n"
    add = ("    from hospital_registry import discover_hospitals\n"
           "    # evaluate.py hard-codes the old hospital names; use this partitioning's own (as the friend's evaluate_run)\n"
           "    evaluate.HOSPITAL_NAMES = discover_hospitals(cfg[\"data\"][\"partitions_root\"])\n")
    if add not in s:
        if s.count(marker) != 1: sys.exit(f"{e}: marker count {s.count(marker)}")
        s = s.replace(marker, marker + add)
        open(e, "w", encoding="utf-8").write(s)
    print("ok", name)
# FedStrag: start from the seed-42 model built in-process (as the other five) instead of the old 13-input novelty file
c = os.path.join(BASE, MODELS["fedstrag"], "configs", "experiment_config.yaml")
t = open(c, encoding="utf-8").read()
t2 = re.sub(r"^initialisation:.*?(?=^\S)", "initialisation:                 # 2026-11-21: seed-42 model built in-process, as the other five models (the old\n  mode: seed                    # novelty initial file has the 13-input metadata layer and cannot be used)\n\n", t, count=1, flags=re.M | re.S)
if t2 == t: sys.exit("fedstrag initialisation not replaced")
open(c, "w", encoding="utf-8").write(t2)
r = os.path.join(BASE, MODELS["fedstrag"], "runtime", "run_fedstrag.py")
s = open(r, encoding="utf-8").read()
old1 = '    if not init.get("file") or init.get("param_hash") != NOVELTY_INITIAL_HASH:'
new1 = '    if init.get("mode") != "seed" and (not init.get("file") or init.get("param_hash") != NOVELTY_INITIAL_HASH):'
start = s.index('    with open(os.path.join(ROOT, cfg["initialisation"]["file"]), "rb") as fh:')
end = s.index("    del saved\n", start) + len("    del saved\n")
block = "".join("    " + line if line.strip() else line for line in s[start:end].splitlines(True))
s = s[:start] + '    if cfg["initialisation"].get("mode") != "seed":   # 2026-11-21: seed mode keeps the seed-42 model built above\n' + block + s[end:]
if s.count(old1) != 1: sys.exit("run_fedstrag: init check pattern not found")
s = s.replace(old1, new1)
open(r, "w", encoding="utf-8").write(s)
print("ok fedstrag init")

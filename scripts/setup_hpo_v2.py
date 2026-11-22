"""Novelty hyperparameter tuning v2 (2026-11-21, user: include ALL parameters of our algorithm). Copy-only patches.

Tuned (19; modality_weight is tied to fitz_weight so the two diversity axes keep their 0.75 total):
  local training : learning_rate, weight_decay, batch_size, local_epochs, dropout, focal_gamma
  federation     : clients_per_round
  selection      : tau (Boltzmann temperature), gamma (diversity amplification), lambda (exposure attenuation),
                   rho (reliability EMA decay)
  unified        : delta (staleness discount, set in selector, straggler manager and aggregation together)
  aggregation    : agg_gamma (diversity amplification in the aggregation weights)
  straggler      : s_max, quorum_frac, mad_k, w_min, history_len
  diversity      : fitz_weight (modality_weight = 0.75 - fitz_weight)
Fixed: 10 hospitals (data), backbone/architecture, optimizer (AdamW), simulated straggler environment, seed.
Validation split / cache from setup_hpo.py are reused unchanged.
"""
import json
import numpy as np
CODE = "/workspace/novelty_hpo/code"
DEF = {"learning_rate": 1e-4, "weight_decay": 1e-4, "batch_size": 20, "local_epochs": 5, "dropout": 0.3,
       "focal_gamma": 2.0, "clients_per_round": 6, "tau": 0.2, "gamma": 1.0, "lambda": 0.05, "rho": 0.85,
       "delta": 0.9, "agg_gamma": 1.0, "s_max": 3, "quorum_frac": 0.7, "mad_k": 2.5, "w_min": 2.0,
       "history_len": 10, "fitz_weight": 0.40}
GRID = {"learning_rate": [5e-5, 1e-4, 2e-4, 3e-4], "weight_decay": [1e-5, 1e-4, 1e-3], "batch_size": [16, 20, 24],
        "local_epochs": [3, 4, 5], "dropout": [0.2, 0.3, 0.5], "focal_gamma": [1.0, 2.0, 3.0],
        "clients_per_round": [5, 6, 7], "tau": [0.1, 0.2, 0.35, 0.5], "gamma": [0.5, 1.0, 1.5, 2.0],
        "lambda": [0.0, 0.02, 0.05, 0.1], "rho": [0.7, 0.85, 0.95], "delta": [0.8, 0.9, 0.95],
        "agg_gamma": [0.5, 1.0, 1.5, 2.0], "s_max": [2, 3, 5], "quorum_frac": [0.6, 0.7, 0.8],
        "mad_k": [2.0, 2.5, 3.0], "w_min": [1.0, 2.0, 4.0], "history_len": [5, 10, 20], "fitz_weight": [0.25, 0.40, 0.55]}
assert set(GRID) == set(DEF)
def full(c): c = dict(c); c["modality_weight"] = round(0.75 - c["fitz_weight"], 2); return c
trials = {"H00": full(DEF)}
rng = np.random.default_rng(2026); seen = {json.dumps(DEF, sort_keys=True)}
while len(trials) < 12:
    c = {k: (v[rng.integers(len(v))]) for k, v in GRID.items()}
    c = {k: (int(x) if isinstance(DEF[k], int) else float(x)) for k, x in c.items()}
    s = json.dumps(c, sort_keys=True)
    if s in seen: continue
    seen.add(s); trials["H%02d" % len(trials)] = full(c)
json.dump(trials, open(CODE + "/ablation/hpo_trials.json", "w"), indent=1)
for k, v in trials.items(): print(k, json.dumps(v))

reg = CODE + "/ablation/experiments.yaml"; t = open(reg).read()
assert "  H00:" not in t and t.count("experiments:\n") == 1
open(reg, "w").write(t.replace("experiments:\n", "experiments:\n" + "".join(
    "  %s: {focal_gamma: %s}\n" % (k, v["focal_gamma"]) for k, v in trials.items())))

cm = CODE + "/ablation/common.py"; t = open(cm).read()
old = '    cfg["local_training"]["focal_gamma"] = float(variant["focal_gamma"])\n'
assert t.count(old) == 1
t = t.replace(old, old + (
    '    _hpo_path = os.path.join(ROOT, "ablation", "hpo_trials.json")\n'
    '    _hpo = json.load(open(_hpo_path)).get(exp_id) if os.path.exists(_hpo_path) else None\n'
    '    if _hpo:   # hyperparameter tuning v2 (2026-11-21); server-side constants are set in run_one.py\n'
    '        cfg["local_training"]["learning_rate"] = float(_hpo["learning_rate"])\n'
    '        cfg["local_training"]["weight_decay"] = float(_hpo["weight_decay"])\n'
    '        cfg["local_training"]["batch_size"] = int(_hpo["batch_size"])\n'
    '        cfg["federation"]["local_epochs"] = int(_hpo["local_epochs"])\n'
    '        cfg["federation"]["clients_per_round"] = int(_hpo["clients_per_round"])\n'
    '        assert float(_hpo["focal_gamma"]) == cfg["local_training"]["focal_gamma"]\n'
    '        cfg["hpo"] = dict(_hpo)\n'))
if "\nimport json" not in t and not t.startswith("import json"): t = "import json\n" + t
open(cm, "w").write(t)

ro = CODE + "/ablation/run_one.py"; t = open(ro).read()
a = '    rid = cfg["experiment"]["run_id"]\n'
assert t.count(a) == 1
t = t.replace(a, a + '    hpo = cfg.get("hpo")\n'
                     '    if hpo:\n'
                     '        os.environ["FEDMEL_DROPOUT"] = str(float(hpo["dropout"]))   # read by models/model.py\n')
b = '        components = reference_server_fn(context)\n'
assert t.count(b) == 1
t = t.replace(b, (
    '        if hpo:\n'
    '            import server.strategy as _sm\n'
    '            if not getattr(_sm, "_hpo_wrapped", False):\n'
    '                _orig = _sm.compute_normalized_diversity_priors\n'
    '                def _div(*a, **k):\n'
    '                    k.setdefault("fitz_weight", float(hpo["fitz_weight"]))\n'
    '                    k.setdefault("modality_weight", float(hpo["modality_weight"]))\n'
    '                    return _orig(*a, **k)\n'
    '                _sm.compute_normalized_diversity_priors = _div\n'
    '                _sm._hpo_wrapped = True\n'
    '            _sm.AGGREGATION_DELTA = float(hpo["delta"])\n'
    '            _sm.AGGREGATION_GAMMA = float(hpo["agg_gamma"])\n') + b)
c = '        apply_variant(components.strategy, ablation["variant"], ablation["harness"], args.seed)\n'
assert t.count(c) == 1
t = t.replace(c, c + (
    '        if hpo:\n'
    '            from collections import deque\n'
    '            import server.strategy as _sm\n'
    '            s = components.strategy._selector\n'
    '            s.tau, s.gamma, s.lambda_, s.rho, s.delta = (float(hpo["tau"]), float(hpo["gamma"]),\n'
    '                float(hpo["lambda"]), float(hpo["rho"]), float(hpo["delta"]))\n'
    '            m = components.strategy._straggler_manager\n'
    '            m.delta, m.s_max, m.quorum_frac, m.mad_k, m.w_min = (float(hpo["delta"]), int(hpo["s_max"]),\n'
    '                float(hpo["quorum_frac"]), float(hpo["mad_k"]), float(hpo["w_min"]))\n'
    '            m._latency_history = deque(maxlen=int(hpo["history_len"]))\n'
    '            import torch.nn as _nn\n'
    '            components.strategy._logger.log_audit({"event": "hpo_override", "requested": hpo,\n'
    '                "applied_selector": [s.tau, s.gamma, s.lambda_, s.rho, s.delta],\n'
    '                "applied_straggler": [m.delta, m.s_max, m.quorum_frac, m.mad_k, m.w_min, m._latency_history.maxlen],\n'
    '                "applied_aggregation": [_sm.AGGREGATION_DELTA, _sm.AGGREGATION_GAMMA],\n'
    '                "applied_n_select": components.strategy._n_select, "dropout_env": os.environ.get("FEDMEL_DROPOUT"),\n'
    '                "diversity_priors_D": {c.client_id: c.D for c in components.strategy._clients.values()},\n'
    '                "selector_class": type(s).__name__, "straggler_class": type(m).__name__})\n'))
open(ro, "w").write(t)

mm = CODE + "/models/model.py"; t = open(mm).read()
d = "nn.Dropout(0.3)"
assert t.count(d) == 1
open(mm, "w").write(t.replace(d, 'nn.Dropout(float(__import__("os").environ.get("FEDMEL_DROPOUT", "0.3")))'))
print("SETUP2 DONE")

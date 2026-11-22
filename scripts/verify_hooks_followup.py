import glob, json
bad = 0
for f in sorted(glob.glob("/workspace/novelty_hpo/code/logs/audit/F*.jsonl")):
    for line in open(f):
        d = json.loads(line)
        if d.get("event") != "hpo_override": continue
        h = d["requested"]
        ok = (d["applied_selector"] == [h["tau"], h["gamma"], h["lambda"], h["rho"], h["delta"]]
              and d["applied_straggler"] == [h["delta"], h["s_max"], h["quorum_frac"], h["mad_k"], h["w_min"], h["history_len"]]
              and d["applied_aggregation"] == [h["delta"], h["agg_gamma"]]
              and d["applied_n_select"] == h["clients_per_round"] and float(d["dropout_env"]) == h["dropout"])
        bad += not ok
        D = d["diversity_priors_D"]; k = sorted(D)[:3]
        print(f.split("/")[-1][:3], "OK" if ok else "MISMATCH", "fitz", h["fitz_weight"], "D", [round(D[x], 4) for x in k], d["selector_class"], d["straggler_class"])
print("mismatches", bad)

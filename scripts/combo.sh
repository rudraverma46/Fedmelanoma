#!/bin/bash
# Combined follow-up: C00 (fresh default control) and C01 (default + agg gamma 0.5 + MAD k 2.0) x seeds 42/43/44,
# 6 runs at once, stage-2 protocol (full train, 30 rounds, real test).
L=/workspace/novelty_hpo/logs
cd /workspace/novelty_hpo/code || exit 1
export FEDMEL_DATA=/workspace/hospital_two_feature/fed_ready/3_Dirichlet_GMM FEDMEL_CACHE=/workspace/six_models_dgmm/cache_dgmm
export FEDMEL_RAW_DIR=/workspace/novelty_hpo/raw_combo FEDMEL_ROUNDS=30 FEDMEL_WORKERS=2 FEDMEL_EXECUTION_ID=hpo_combo
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
python3 -c "from ablation.common import build_run_config as b; c=b('C01',42); print(c['experiment']['run_id'], c['hpo'])" > $L/combo_precheck.log 2>&1 || { echo "$(date '+%F %T') PRECHECK FAILED" >> $L/combo.log; exit 1; }
echo "$(date '+%F %T') combo start: C00 C01 x seeds 42 43 44" >> $L/combo.log
for t in C00 C01; do for s in 42 43 44; do
  nohup python3 -m ablation.worker --gpu 0 --runs $t:$s > $L/co_"$t"_"$s".log 2>&1 &
done; done
wait
echo "$(date '+%F %T') combo ALL DONE" >> $L/combo.log

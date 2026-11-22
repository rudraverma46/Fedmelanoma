#!/bin/bash
# Stage 2: retrain the 2 best NON-default configurations (validation ranking) on the FULL training data, 30 rounds,
# real test set -- same protocol as the ablation E00_full run (default configuration), which is the comparison arm.
L=/workspace/novelty_hpo/logs
until grep -q "stage1 ALL DONE" $L/stage1.log 2>/dev/null; do sleep 60; done
cd /workspace/novelty_hpo && python3 select_trials.py > $L/selection.log 2>&1 || { echo "$(date '+%F %T') SELECTION FAILED" >> $L/stage2.log; exit 1; }
TOP=$(python3 -c "import json; print(' '.join(json.load(open('$L/selection.json'))['top2_non_default']))")
cd /workspace/novelty_hpo/code || exit 1
export FEDMEL_DATA=/workspace/hospital_two_feature/fed_ready/3_Dirichlet_GMM FEDMEL_CACHE=/workspace/six_models_dgmm/cache_dgmm
export FEDMEL_RAW_DIR=/workspace/novelty_hpo/raw_stage2 FEDMEL_ROUNDS=30 FEDMEL_WORKERS=3 FEDMEL_EXECUTION_ID=hpo_stage2
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
echo "$(date '+%F %T') stage2 start: $TOP" >> $L/stage2.log
for t in $TOP; do nohup python3 -m ablation.worker --gpu 0 --runs $t:42 > $L/s2_$t.log 2>&1 & done
wait
echo "$(date '+%F %T') stage2 ALL DONE" >> $L/stage2.log

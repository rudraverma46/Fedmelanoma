#!/bin/bash
# Stage 2 (v3): the 2 best NON-default configurations (mean validation ranking over seeds 42-44) AND the default H00,
# each retrained on the FULL training data for 30 rounds at seeds 42, 43, 44, evaluated on the real test set.
# H00 is rerun because the ablation E00_full exists for seed 42 only; E00_full s42 stays as a sanity check.
L=/workspace/novelty_hpo/logs
until grep -q "stage1 seeds ALL DONE" $L/stage1.log 2>/dev/null; do sleep 60; done
cd /workspace/novelty_hpo && python3 select_trials_v3.py > $L/selection.log 2>&1 || { echo "$(date '+%F %T') SELECTION FAILED" >> $L/stage2.log; exit 1; }
TOP=$(python3 -c "import json; print(' '.join(json.load(open('$L/selection.json'))['top2_non_default']))")
cd /workspace/novelty_hpo/code || exit 1
export FEDMEL_DATA=/workspace/hospital_two_feature/fed_ready/3_Dirichlet_GMM FEDMEL_CACHE=/workspace/six_models_dgmm/cache_dgmm
export FEDMEL_RAW_DIR=/workspace/novelty_hpo/raw_stage2 FEDMEL_ROUNDS=30 FEDMEL_WORKERS=2 FEDMEL_EXECUTION_ID=hpo_stage2_v3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
echo "$(date '+%F %T') stage2 start: $TOP H00 x seeds 42 43 44" >> $L/stage2.log
for t in $TOP H00; do for s in 42 43 44; do
  nohup python3 -m ablation.worker --gpu 0 --runs $t:$s > $L/s2_"$t"_"$s".log 2>&1 &
done; done
wait
echo "$(date '+%F %T') stage2 ALL DONE" >> $L/stage2.log

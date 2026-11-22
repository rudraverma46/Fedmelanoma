#!/bin/bash
# Stage 1, seeds 43 and 44 (user 2026-11-21: "also use seeds 43 and 44 with 42"). Same 12 configurations, same
# 12 rounds, same validation split; starts when the seed-42 wave finishes. 24 runs, 12 at a time (GPU limit), seed-major.
L=/workspace/novelty_hpo/logs
until grep -q "stage1 ALL DONE" $L/stage1.log 2>/dev/null; do sleep 60; done
export FEDMEL_DATA=/workspace/novelty_hpo/data_val FEDMEL_CACHE=/workspace/novelty_hpo/cache_val
export FEDMEL_RAW_DIR=/workspace/novelty_hpo/raw_stage1 FEDMEL_ROUNDS=12 FEDMEL_WORKERS=2 FEDMEL_EXECUTION_ID=hpo_stage1_v3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
echo "$(date '+%F %T') stage1 seeds 43/44 start" >> $L/stage1.log
for s in 43 44; do for t in H00 H01 H02 H03 H04 H05 H06 H07 H08 H09 H10 H11; do echo $t:$s; done; done \
  | xargs -P 12 -n 1 /workspace/novelty_hpo/run_trial.sh
echo "$(date '+%F %T') stage1 seeds ALL DONE" >> $L/stage1.log

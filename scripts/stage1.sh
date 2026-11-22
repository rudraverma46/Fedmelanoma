#!/bin/bash
# Stage 1 (v2, all 19 algorithm hyperparameters): 12 configurations (H00 default + 11 random), 12 rounds,
# train = 85% of training patients, evaluated each round on the 15% validation patients. 12 parallel = GPU memory limit.
cd /workspace/novelty_hpo/code || exit 1
export FEDMEL_DATA=/workspace/novelty_hpo/data_val FEDMEL_CACHE=/workspace/novelty_hpo/cache_val
export FEDMEL_RAW_DIR=/workspace/novelty_hpo/raw_stage1 FEDMEL_ROUNDS=12 FEDMEL_WORKERS=2 FEDMEL_EXECUTION_ID=hpo_stage1_v2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
L=/workspace/novelty_hpo/logs
python3 -c "from ablation.common import build_run_config as b; c=b('H04',42); print(c['local_training'], c['federation'], c['hpo'], c['data'], c['experiment']['run_id'])" > $L/precheck.log 2>&1 || { echo "$(date '+%F %T') PRECHECK FAILED" >> $L/stage1.log; exit 1; }
echo "$(date '+%F %T') stage1 v2 start" >> $L/stage1.log
for t in H00 H01 H02 H03 H04 H05 H06 H07 H08 H09 H10 H11; do
  nohup python3 -m ablation.worker --gpu 0 --runs $t:42 > $L/s1_$t.log 2>&1 &
done
wait
echo "$(date '+%F %T') stage1 ALL DONE" >> $L/stage1.log

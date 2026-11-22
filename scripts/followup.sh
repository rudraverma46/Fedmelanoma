#!/bin/bash
# Follow-up: F01-F04 x seeds 42/43/44 (12 runs, all at once = GPU limit), stage-2 protocol (full train, 30 rounds, real test).
# The comparison arm is the existing stage-2 H00 (raw_stage2/H00, seeds 42-44), linked into raw_followup for the report.
L=/workspace/novelty_hpo/logs
cd /workspace/novelty_hpo/code || exit 1
export FEDMEL_DATA=/workspace/hospital_two_feature/fed_ready/3_Dirichlet_GMM FEDMEL_CACHE=/workspace/six_models_dgmm/cache_dgmm
export FEDMEL_RAW_DIR=/workspace/novelty_hpo/raw_followup FEDMEL_ROUNDS=30 FEDMEL_WORKERS=2 FEDMEL_EXECUTION_ID=hpo_followup
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
python3 -c "from ablation.common import build_run_config as b; c=b('F03',42); print(c['experiment']['run_id'], c['hpo'])" > $L/followup_precheck.log 2>&1 || { echo "$(date '+%F %T') PRECHECK FAILED" >> $L/followup.log; exit 1; }
echo "$(date '+%F %T') followup start: F01 F02 F03 F04 x seeds 42 43 44" >> $L/followup.log
for t in F01 F02 F03 F04; do for s in 42 43 44; do
  nohup python3 -m ablation.worker --gpu 0 --runs $t:$s > $L/fu_"$t"_"$s".log 2>&1 &
done; done
wait
echo "$(date '+%F %T') followup ALL DONE" >> $L/followup.log

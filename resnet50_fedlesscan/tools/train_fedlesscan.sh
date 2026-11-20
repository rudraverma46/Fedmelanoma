#!/bin/bash
# FedLesScan run: full training, then final evaluation of round 50 and of the best checkpoint,
# leakage analysis and the report data. One run, on the GPU.
exec >> "/d/MAIN BACKUP/backup 3/Fedmelanoma/resnet50_fedlesscan_novelty_env/logs/train_fedlesscan.log" 2>&1
BASE="/d/MAIN BACKUP/backup 3/Fedmelanoma"
PY="$BASE/FedAvg_RESNET 50/.venv/Scripts/python.exe"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
run="resnet50-fedlesscan-novelty-env-seed42"
cd "$BASE/resnet50_fedlesscan_novelty_env" || exit 1
echo "$(date '+%F %T') START $run"
"$PY" tools/run_supervised.py "$run" || { echo "$(date '+%F %T') TRAINING STOPPED $run"; exit 2; }
echo "$(date '+%F %T') TRAINED $run"
"$PY" evaluation/evaluate_checkpoint.py "checkpoints/$run/global_round_0050.pth" "results/$run/final_round50" > "logs/final_round50.log" 2>&1 || { echo "EVAL FAILED round50"; exit 3; }
"$PY" evaluation/evaluate_checkpoint.py "checkpoints/$run/best_global_model.pth" "results/$run/final_best" > "logs/final_best.log" 2>&1 || { echo "EVAL FAILED best"; exit 3; }
"$PY" evaluation/leakage_analysis.py "$run" > "logs/leakage.log" 2>&1 || { echo "LEAKAGE FAILED"; exit 3; }
"$PY" evaluation/build_report.py "$run" > "logs/build_report.log" 2>&1 || { echo "REPORT FAILED"; exit 3; }
echo "$(date '+%F %T') DONE $run"

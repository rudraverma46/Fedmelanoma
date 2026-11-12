#!/bin/bash
# One run at a time: full training, then final evaluation, for textbook FedAvg and then textbook FedProx.
exec >> "/d/MAIN BACKUP/backup 3/Fedmelanoma/resnet50_fedavg_textbook/logs/train_both.log" 2>&1
BASE="/d/MAIN BACKUP/backup 3/Fedmelanoma"
PY="$BASE/FedAvg_RESNET 50/.venv/Scripts/python.exe"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
for item in "resnet50_fedavg_textbook:resnet50-fedavg-textbook-seed42" "resnet50_fedprox_textbook:resnet50-fedprox-textbook-mu0.01-seed42"; do
  folder="${item%%:*}"; run="${item##*:}"
  cd "$BASE/$folder" || exit 1
  echo "$(date '+%F %T') START $run"
  "$PY" tools/run_supervised.py "$run" || { echo "$(date '+%F %T') TRAINING STOPPED $run"; exit 2; }
  echo "$(date '+%F %T') TRAINED $run"
  "$PY" evaluation/evaluate_checkpoint.py "checkpoints/$run/best_global_model.pth" "results/$run/final_best" > "logs/final_best.log" 2>&1 || { echo "EVAL FAILED best $run"; exit 3; }
  "$PY" evaluation/evaluate_checkpoint.py "checkpoints/$run/global_round_0050.pth" "results/$run/final_round50" > "logs/final_round50.log" 2>&1 || { echo "EVAL FAILED round50 $run"; exit 3; }
  "$PY" evaluation/leakage_analysis.py "$run" > "logs/leakage.log" 2>&1 || { echo "LEAKAGE FAILED $run"; exit 3; }
  "$PY" evaluation/build_report.py "$run" > "logs/build_report.log" 2>&1 || { echo "REPORT FAILED $run"; exit 3; }
  echo "$(date '+%F %T') DONE $run"
done
echo "$(date '+%F %T') ALL DONE"

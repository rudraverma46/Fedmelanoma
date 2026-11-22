#!/bin/bash
# fedasync run: full training (resumes after a crash), then the final evaluation of the final and the
# best checkpoint on the pooled test set. One run, on the GPU.
exec >> "/d/MAIN BACKUP/backup 3/Fedmelanoma/resnet50_fedasync/logs/train_fedasync.log" 2>&1
BASE="/d/MAIN BACKUP/backup 3/Fedmelanoma"
PY="$BASE/FedAvg_RESNET 50/.venv/Scripts/python.exe"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
run="resnet50-fedasync-seed42"
cd "$BASE/resnet50_fedasync" || exit 1
mkdir -p logs
echo "$(date '+%F %T') START $run"
"$PY" tools/run_supervised.py "$run" || { echo "$(date '+%F %T') TRAINING STOPPED $run"; exit 2; }
echo "$(date '+%F %T') TRAINED $run"
"$PY" evaluation/evaluate_checkpoint.py "checkpoints/$run/final_global_model.pth" "results/$run/final" > "logs/final.log" 2>&1 || { echo "EVAL FAILED final"; exit 3; }
"$PY" evaluation/evaluate_checkpoint.py "checkpoints/$run/best_global_model.pth" "results/$run/final_best" > "logs/final_best.log" 2>&1 || { echo "EVAL FAILED best"; exit 3; }
echo "$(date '+%F %T') DONE $run"

#!/usr/bin/env bash
# FedStrag on a remote Linux GPU server (same layout the FedLesScan bundle used).
#
# Expected layout (folders side by side):
#   <base>/resnet_50_fedstrag_final/                         this folder
#   <base>/without_federated/client/hospital_partitions/     the 10 hospital folders (metadata_*.csv; images optional)
#   <base>/resnet 50_novelty_fedavg/cache/                   the 40 cache files (.npy + .json)
#
# Usage, from a terminal in this folder (Jupyter: File > New > Terminal):
#   bash tools/remote_run.sh check     layout, GPU, Python packages, CPU tests             (minutes)
#   bash tools/remote_run.sh verify    all tests + GPU smoke run with stop/resume          (~10-20 min)
#   bash tools/remote_run.sh full      the 50-iteration experiment + final evaluations     (hours)
# For `full`, survive a closed browser tab:
#   nohup bash tools/remote_run.sh full > logs/full_run.out 2>&1 &
#   tail -f logs/full_run.out
# Stop cleanly after the current iteration:  touch logs/STOP     Resume: run `full` again (it resumes).
set -u
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
RUN="resnet50-fedstrag-final-seed42"
mkdir -p logs
stamp() { date '+%F %T'; }

check() {
  local ok=0
  [ -d "../without_federated/client/hospital_partitions" ] || { echo "MISSING ../without_federated/client/hospital_partitions"; ok=1; }
  [ -d "../resnet 50_novelty_fedavg/cache" ] || { echo "MISSING ../resnet 50_novelty_fedavg/cache"; ok=1; }
  [ "$ok" = 0 ] || return 1
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || return 1
  "$PY" -c "import sys, torch, torchvision, flwr, numpy, pandas, sklearn, cv2, yaml, psutil; \
print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu ok', torch.cuda.is_available())" || return 1
  echo "pinned versions (requirements-lock.txt):"; grep -iE "^(torch|torchvision|flwr|numpy)==" requirements-lock.txt
  df -h . | tail -1
  "$PY" -W ignore -m pytest tests/test_00_reference_integrity.py tests/test_02_fedstrag_math.py tests/test_04_environment.py \
        -p no:cacheprovider -q --tb=short
}

case "${1:-}" in
  check)  check ;;
  verify) check && "$PY" -W ignore tools/run_verification.py ;;
  full)
    echo "$(stamp) START $RUN"
    "$PY" tools/run_supervised.py "$RUN" || { echo "$(stamp) TRAINING STOPPED $RUN (see logs/console/)"; exit 2; }
    echo "$(stamp) TRAINED $RUN"
    # PRIMARY reported result: the iteration-50 model (no test-based selection)
    "$PY" -W ignore evaluation/evaluate_checkpoint.py "checkpoints/$RUN/final_global_model.pth" "results/$RUN/final" > logs/final.log 2>&1 \
      || { echo "EVAL FAILED final"; exit 3; }
    # SECONDARY diagnostic only: chosen by ROC-AUC on local TEST partitions, never the reported result
    "$PY" -W ignore evaluation/evaluate_checkpoint.py "checkpoints/$RUN/best_global_model.pth" "results/$RUN/secondary_best_test_selected" > logs/secondary_best.log 2>&1 \
      || { echo "EVAL FAILED secondary best"; exit 3; }
    echo "$(stamp) DONE $RUN: PRIMARY = results/$RUN/final/metrics.json (iteration 50); summary results/$RUN/run_summary.json"
    ;;
  *) echo "usage: bash tools/remote_run.sh check|verify|full"; exit 1 ;;
esac

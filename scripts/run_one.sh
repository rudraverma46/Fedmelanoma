#!/bin/bash
# One model: train (supervised, resumes after crashes), then evaluate the final and the best checkpoint on the pooled test set.
# usage: run_one.sh <name> <model dir> <final checkpoint file name>
name="$1"; dir="$2"; final="$3"; run="$name-dgmm-seed42"
L=/workspace/six_models_dgmm/logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
cd "$dir" || exit 1
st() { echo "$(date '+%F %T') $*" >> "$L/$name.status"; }
st "START $run"
python3 tools/run_supervised.py "$run" >> "$L/$name.train.log" 2>&1 || { st "TRAINING STOPPED (exit $?)"; exit 2; }
st TRAINED
python3 evaluation/evaluate_checkpoint.py "checkpoints/$run/$final" "results/$run/final" > "$L/$name.eval_final.log" 2>&1 || { st "EVAL FAILED final"; exit 3; }
python3 evaluation/evaluate_checkpoint.py "checkpoints/$run/best_global_model.pth" "results/$run/best" > "$L/$name.eval_best.log" 2>&1 || st "EVAL FAILED best"
st DONE

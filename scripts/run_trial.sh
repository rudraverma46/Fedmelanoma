#!/bin/bash
# one stage-1 trial for one seed, e.g. run_trial.sh H03:43 (environment comes from stage1_seeds.sh)
cd /workspace/novelty_hpo/code || exit 1
python3 -m ablation.worker --gpu 0 --runs "$1" > /workspace/novelty_hpo/logs/s1_$(echo "$1" | tr : _).log 2>&1

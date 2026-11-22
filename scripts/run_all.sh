#!/bin/bash
# Launch the six models at once (user decision 2026-11-21: parallel, 15 CPU cores -> 2 threads each) + checkpoint pruner.
B=/workspace/six_models_dgmm
cd "$B" || exit 1
mkdir -p logs
launch() { setsid nohup bash "$B/run_one.sh" "$1" "$B/$2" "$3" > "$B/logs/$1.launcher.log" 2>&1 < /dev/null & }
launch fedavg     "src_resnet50-fedavg-textbook/resnet50_fedavg_textbook"   global_round_0050.pth
launch fedprox    "src_resnet50-fedprox-textbook/resnet50_fedprox_textbook" global_round_0050.pth
launch novelty    "src_resnet50-no-confirm-type/RESNET50_NOVELTY MODEL"     global_round_0050.pth
launch fedlesscan "src_resnet_50_fedlesscan/resnet50_fedlesscan"            global_round_0050.pth
launch fedasync   "src_resnet50-fedasync/resnet50_fedasync"                 final_global_model.pth
launch fedstrag   "src_resnet_50_fedstrag"                                  final_global_model.pth
setsid nohup python3 "$B/prune_checkpoints.py" > "$B/logs/prune.log" 2>&1 < /dev/null &
echo launched

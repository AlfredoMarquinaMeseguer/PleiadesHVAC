#!/bin/bash

set -e
# "transformer"
MODELS=( "lstm" "convlstm" "gru" "gruSimple")

for model in "${MODELS[@]}"; do
    echo "========================================="
    echo "Running model: $model"
    echo "========================================="

    # --- First program (HFL) ---
    echo "[HFL] Starting run for model: $model"
    flwr run . --stream --run-config "model-type=\"$model\""

    # Move the 4 JSON outputs to metrics/$model/HFL/
    dest_hfl="metrics/$model/HFL"
    mkdir -p "$dest_hfl"
    echo "[HFL] Moving JSON files to $dest_hfl"
    mv data/metrics/*.json "$dest_hfl/"

    echo "[HFL] Done for model: $model"

    # --- Second program (FL) ---
    echo "[FL] Starting run for model: $model"
    flwr run pleiadesHVAC_edge/ --stream --run-config "model-type=\"$model\" num-server-rounds=3 num-nodes=15"

    # Move the 1 JSON output to metrics/$model/FL/
    dest_fl="metrics/$model/FL"
    mkdir -p "$dest_fl"
    echo "[FL] Moving JSON file to $dest_fl"
    mv data/metrics/*.json "$dest_fl/"

    echo "[FL] Done for model: $model"

    echo "Completed: $model"
    echo ""
done

echo "All models finished."
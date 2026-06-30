#!/bin/bash

set -e
MODELS=("lstm" "convlstm" "gru" "gruSimple" "transformer")

for model in "${MODELS[@]}"; do
    for i in $(seq 1 10); do
        echo "========================================="
        echo "Running model: $model"
        echo "========================================="

        # --- First program (HFL) ---
        echo "[HFL] Starting run for model: $model"
        flwr run . --stream --run-config "model-type=\"$model\""

        # Move the 4 JSON outputs to metrics/$model/HFL/
        dest_hfl="metrics/$model/$i/HFL"
        mkdir -p "$dest_hfl"
        echo "[HFL] Moving JSON files to $dest_hfl"
        cp -r data/metrics/*.json "$dest_hfl/"

        #Move weigths to metrics/$model/HFL/weigths
        weigths_hfl="$dest_hfl/weigths"
        mkdir -p "$weigths_hfl"
        echo "[HFL] Moving weigths to $weigths_hfl"
        cp -r state/* "$weigths_hfl/"

        echo "[HFL] Done for model: $model"
    done

    rm -r state/*

    for i in $(seq 1 10); do
        # --- Second program (FL) ---
        echo "[FL] Starting run for model: $model"
        flwr run pleiadesHVAC_edge/ --stream --run-config "model-type=\"$model\" num-nodes=15"

        # Move the 1 JSON output to metrics/$model/FL/
        dest_fl="metrics/$model/$i/FL"
        mkdir -p "$dest_fl"
        echo "[FL] Moving JSON file to $dest_fl"
        cp -r data/metrics/*.json "$dest_fl/"

        #Move weigths to metrics/$model/HFL/weigths
        weigths_fl="$dest_fl/weigths"
        mkdir -p "$weigths_fl"
        echo "[FL] Moving weigths to $weigths_fl"
        cp -r state/* "$weigths_fl/"

        echo "[FL] Done for model: $model"

        echo "Completed: $model"
        echo ""
    done

    rm -r state/*
done

echo "All models finished."
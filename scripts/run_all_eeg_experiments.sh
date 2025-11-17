#!/bin/bash
# Run all EEG experiments on local GPU hardware
# No SLURM required - runs sequentially

MODELS=("quantum_hydra" "quantum_hydra_hybrid" "quantum_mamba" "quantum_mamba_hybrid" "quantum_mamba_lite" "quantum_mamba_hybrid_lite" "classical_hydra" "classical_mamba")
SEEDS=(2024 2025 2026)

echo "================================================================================"
echo "RUNNING ALL EEG EXPERIMENTS"
echo "================================================================================"
echo "Models: ${MODELS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "Total experiments: $((${#MODELS[@]} * ${#SEEDS[@]}))"
echo "================================================================================"
echo ""

mkdir -p eeg_results/logs

count=0
total=$((${#MODELS[@]} * ${#SEEDS[@]}))

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ((count++))
        echo "[$count/$total] Running: $model (seed=$seed)"

        python experiments/run_single_model_eeg.py \
            --model-name $model \
            --n-qubits 6 \
            --qlcu-layers 2 \
            --d-model 128 \
            --d-state 16 \
            --n-epochs 50 \
            --batch-size 32 \
            --lr 0.001 \
            --sample-size 10 \
            --sampling-freq 100 \
            --early-stopping-patience 10 \
            --seed $seed \
            --output-dir ./eeg_results \
            --device cuda \
            > eeg_results/logs/eeg_${model}_seed${seed}.log 2>&1

        if [ $? -eq 0 ]; then
            echo "  ✓ Completed successfully"
        else
            echo "  ✗ Failed (check log: eeg_results/logs/eeg_${model}_seed${seed}.log)"
        fi
        echo ""
    done
done

echo "================================================================================"
echo "ALL EEG EXPERIMENTS COMPLETE!"
echo "================================================================================"
echo "Results saved to: ./eeg_results/"
echo "To aggregate results, run:"
echo "  python experiments/aggregate_eeg_results.py"
echo "================================================================================"

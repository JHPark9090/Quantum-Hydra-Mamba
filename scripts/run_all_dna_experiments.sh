#!/bin/bash
# Run all DNA experiments on local GPU hardware
# No SLURM required - runs sequentially

MODELS=("quantum_hydra" "quantum_hydra_hybrid" "quantum_mamba" "quantum_mamba_hybrid" "classical_hydra" "classical_mamba")
SEEDS=(2024 2025 2026)

echo "================================================================================"
echo "RUNNING ALL DNA EXPERIMENTS"
echo "================================================================================"
echo "Models: ${MODELS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "Total experiments: $((${#MODELS[@]} * ${#SEEDS[@]}))"
echo "================================================================================"
echo ""

mkdir -p dna_results/logs

count=0
total=$((${#MODELS[@]} * ${#SEEDS[@]}))

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ((count++))
        echo "[$count/$total] Running: $model (seed=$seed)"

        python experiments/run_single_model_dna.py \
            --model-name $model \
            --n-qubits 6 \
            --qlcu-layers 2 \
            --d-model 128 \
            --d-state 16 \
            --n-epochs 50 \
            --batch-size 32 \
            --lr 0.001 \
            --early-stopping-patience 10 \
            --n-train 100 \
            --n-valtest 50 \
            --encoding onehot \
            --seed $seed \
            --output-dir ./dna_results \
            --device cuda \
            > dna_results/logs/dna_${model}_seed${seed}.log 2>&1

        if [ $? -eq 0 ]; then
            echo "  ✓ Completed successfully"
        else
            echo "  ✗ Failed (check log: dna_results/logs/dna_${model}_seed${seed}.log)"
        fi
        echo ""
    done
done

echo "================================================================================"
echo "ALL DNA EXPERIMENTS COMPLETE!"
echo "================================================================================"
echo "Results saved to: ./dna_results/"
echo "To aggregate results, run:"
echo "  python experiments/aggregate_dna_results.py"
echo "================================================================================"

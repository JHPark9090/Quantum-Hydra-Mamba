#!/bin/bash
# Run all MNIST experiments on local GPU hardware
# No SLURM required - runs sequentially

MODELS=("quantum_hydra" "quantum_hydra_hybrid" "quantum_mamba" "quantum_mamba_hybrid" "classical_hydra" "classical_mamba")
SEEDS=(2024 2025 2026)

echo "================================================================================"
echo "RUNNING ALL MNIST EXPERIMENTS"
echo "================================================================================"
echo "Models: ${MODELS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "Total experiments: $((${#MODELS[@]} * ${#SEEDS[@]}))"
echo "================================================================================"
echo ""

mkdir -p mnist_results/logs

count=0
total=$((${#MODELS[@]} * ${#SEEDS[@]}))

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ((count++))
        echo "[$count/$total] Running: $model (seed=$seed)"

        python experiments/run_single_model_mnist.py \
            --model-name $model \
            --n-qubits 6 \
            --qlcu-layers 2 \
            --d-model 128 \
            --d-state 16 \
            --n-epochs 50 \
            --batch-size 32 \
            --lr 0.001 \
            --n-train 500 \
            --n-valtest 250 \
            --early-stopping-patience 10 \
            --seed $seed \
            --output-dir ./mnist_results \
            --device cuda \
            > mnist_results/logs/mnist_${model}_seed${seed}.log 2>&1

        if [ $? -eq 0 ]; then
            echo "  ✓ Completed successfully"
        else
            echo "  ✗ Failed (check log: mnist_results/logs/mnist_${model}_seed${seed}.log)"
        fi
        echo ""
    done
done

echo "================================================================================"
echo "ALL MNIST EXPERIMENTS COMPLETE!"
echo "================================================================================"
echo "Results saved to: ./mnist_results/"
echo "To aggregate results, run:"
echo "  python experiments/aggregate_mnist_results.py"
echo "================================================================================"

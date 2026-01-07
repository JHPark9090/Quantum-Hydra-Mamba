#!/bin/bash
# Run all Forrelation experiments on local GPU hardware
# No SLURM required - runs sequentially

MODELS=("quantum_hydra" "quantum_hydra_hybrid" "quantum_mamba" "quantum_mamba_hybrid" "quantum_mamba_lite" "quantum_mamba_hybrid_lite" "classical_hydra" "classical_mamba")
DATASETS=("forrelation_L20.pt" "forrelation_L40.pt" "forrelation_L80.pt" "forrelation_L160.pt" "forrelation_n8_L40.pt" "forrelation_n8_L80.pt" "forrelation_n8_L160.pt" "forrelation_n8_L320.pt")
SEEDS=(2024 2025 2026)

echo "================================================================================"
echo "RUNNING ALL FORRELATION EXPERIMENTS (QUANTUM ADVANTAGE TEST)"
echo "================================================================================"
echo "Models: ${MODELS[@]}"
echo "Datasets: ${#DATASETS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "Total experiments: $((${#MODELS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]}))"
echo ""
echo "WARNING: This will take a long time! Consider running in screen/tmux."
echo "================================================================================"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

mkdir -p forrelation_results/logs

count=0
total=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]}))

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            ((count++))
            dataset_name="${dataset%.pt}"
            echo "[$count/$total] Running: $model on $dataset_name (seed=$seed)"

            python experiments/run_single_model_forrelation.py \
                --model-name $model \
                --dataset-path forrelation_data/$dataset \
                --n-qubits 6 \
                --qlcu-layers 2 \
                --d-model 128 \
                --d-state 16 \
                --n-epochs 100 \
                --batch-size 32 \
                --lr 0.001 \
                --early-stopping-patience 15 \
                --seed $seed \
                --output-dir ./forrelation_results \
                --device cuda \
                > forrelation_results/logs/forrelation_${model}_${dataset_name}_seed${seed}.log 2>&1

            if [ $? -eq 0 ]; then
                echo "  ✓ Completed successfully"
            else
                echo "  ✗ Failed (check log)"
            fi
            echo ""
        done
    done
done

echo "================================================================================"
echo "ALL FORRELATION EXPERIMENTS COMPLETE!"
echo "================================================================================"
echo "Results saved to: ./forrelation_results/"
echo "To analyze results and test for quantum advantage, run:"
echo "  python experiments/aggregate_forrelation_results.py"
echo "================================================================================"

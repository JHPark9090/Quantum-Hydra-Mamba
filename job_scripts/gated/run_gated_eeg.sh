#!/bin/bash
#SBATCH --job-name=gated_eeg
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-9
#SBATCH --output=/pscratch/sd/j/junghoon/quantum_hydra_mamba/results/gated/logs/gated_eeg_%a.log
#SBATCH --error=/pscratch/sd/j/junghoon/quantum_hydra_mamba/results/gated/logs/gated_eeg_%a.log

# Array job for Quantum Gated Models on EEG
# 2 models × 5 seeds = 10 jobs

cd /pscratch/sd/j/junghoon/quantum_hydra_mamba

# Activate conda environment
source /pscratch/sd/j/junghoon/.bashrc
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

# Define models and seeds
MODELS=(quantum_mamba_gated quantum_hydra_gated)
SEEDS=(2024 2025 2026 2027 2028)

# Calculate model and seed indices
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))

MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=========================================="
echo "Job ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Seed: $SEED"
echo "=========================================="

# Run training (with -u for unbuffered output)
python -u scripts/run_gated_models.py \
    --model-name $MODEL \
    --dataset eeg \
    --n-qubits 6 \
    --qlcu-layers 2 \
    --hidden-dim 64 \
    --chunk-size 8 \
    --n-epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --early-stopping-patience 10 \
    --sample-size 50 \
    --sampling-freq 80 \
    --seed $SEED \
    --output-dir ./results/gated \
    --device cuda

echo "Job completed: $MODEL seed=$SEED"

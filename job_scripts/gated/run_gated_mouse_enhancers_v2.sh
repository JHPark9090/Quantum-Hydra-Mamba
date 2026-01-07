#!/bin/bash
#SBATCH --job-name=gated_mouse_v2
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-9
#SBATCH --output=/pscratch/sd/j/junghoon/quantum_hydra_mamba/results/genomic_v2/logs/gated_mouse_enhancers_%a.log
#SBATCH --error=/pscratch/sd/j/junghoon/quantum_hydra_mamba/results/genomic_v2/logs/gated_mouse_enhancers_%a.log

# Array job for Quantum Gated Models on Mouse Enhancers
# VERSION 2: Using nn.Linear instead of Conv1d
# 2 models × 5 seeds = 10 jobs
# Dataset: ~1K samples, 4776 bp sequences (LONG sequences)
# Extended time limit (48h) due to long sequences

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
echo "Dataset: dummy_mouse_enhancers_ensembl"
echo "Version: v2 (nn.Linear)"
echo "=========================================="

# Run training
# dummy_mouse_enhancers_ensembl: ~1K samples, 4776 bp
# Using chunk_size=64 for long 4776 timesteps
# Smaller batch size due to long sequences
python -u scripts/run_gated_genomic.py \
    --model-name $MODEL \
    --dataset dummy_mouse_enhancers_ensembl \
    --n-qubits 6 \
    --qlcu-layers 2 \
    --hidden-dim 64 \
    --chunk-size 64 \
    --n-epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --early-stopping-patience 15 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed $SEED \
    --output-dir ./results/genomic \
    --device cuda

echo "Job completed: $MODEL seed=$SEED"

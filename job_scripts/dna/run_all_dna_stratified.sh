#!/bin/bash
#SBATCH --job-name=dna_stratified
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-29
#SBATCH --output=/pscratch/sd/j/junghoon/results/dna_results/logs/dna_%A_%a.log
#SBATCH --error=/pscratch/sd/j/junghoon/results/dna_results/logs/dna_%A_%a.log

# Array job for all DNA experiments with STRATIFIED SAMPLING
# 6 models × 5 seeds = 30 jobs
#
# Models: quantum_hydra, quantum_hydra_hybrid, quantum_mamba,
#         quantum_mamba_hybrid, classical_hydra, classical_mamba
# Seeds: 2024, 2025, 2026, 2027, 2028
#
# Date: November 21, 2025

# Move to project directory
cd /pscratch/sd/j/junghoon

# Activate conda environment
source /pscratch/sd/j/junghoon/.bashrc
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

# Change to project directory
cd /pscratch/sd/j/junghoon/quantum_hydra_mamba

# Model and seed mapping
# Array index = model_idx * 5 + seed_idx
MODELS=(quantum_hydra quantum_hydra_hybrid quantum_mamba quantum_mamba_hybrid classical_hydra classical_mamba)
SEEDS=(2024 2025 2026 2027 2028)

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))

MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "============================================================"
echo "DNA Classification - Stratified Sampling"
echo "============================================================"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Seed: $SEED"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L | head -1)"
echo "============================================================"
echo ""

# Run training
python scripts/run_single_model_dna.py \
    --model-name $MODEL \
    --n-qubits 6 \
    --qlcu-layers 2 \
    --d-model 128 \
    --d-state 16 \
    --n-epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --early-stopping-patience 10 \
    --n-train 70 \
    --n-valtest 36 \
    --encoding onehot \
    --seed $SEED \
    --output-dir ./results/dna_results \
    --device cuda

echo ""
echo "============================================================"
echo "Job completed: $MODEL seed=$SEED"
echo "End time: $(date)"
echo "============================================================"

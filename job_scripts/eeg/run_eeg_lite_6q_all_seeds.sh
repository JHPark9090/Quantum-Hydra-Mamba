#!/bin/bash
#SBATCH --job-name=eeg_lite_6q
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-9
#SBATCH --output=/pscratch/sd/j/junghoon/results/eeg_results/logs/eeg_lite_6q_%A_%a.log
#SBATCH --error=/pscratch/sd/j/junghoon/results/eeg_results/logs/eeg_lite_6q_%A_%a.log

# Array job for 6-qubit EEG Lite experiments
# Array indices 0-9: 2 models × 5 seeds
#
# Index mapping:
#   0-4: quantum_mamba_lite (seeds 2024-2028)
#   5-9: quantum_mamba_hybrid_lite (seeds 2024-2028)
#
# Single GPU configuration (optimal for quantum models)
# Resubmission with memory-optimized Lite models (vectorized batch processing)
# Time increased to 24h due to MNE lock contention during resampling
# Date: November 21, 2025

# Move to project directory
cd /pscratch/sd/j/junghoon

# Activate conda environment
source /pscratch/sd/j/junghoon/.bashrc
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

# Change to project directory
cd /pscratch/sd/j/junghoon/quantum_hydra_mamba

# Model and seed mapping
MODELS=(
    "quantum_mamba_lite"
    "quantum_mamba_lite"
    "quantum_mamba_lite"
    "quantum_mamba_lite"
    "quantum_mamba_lite"
    "quantum_mamba_hybrid_lite"
    "quantum_mamba_hybrid_lite"
    "quantum_mamba_hybrid_lite"
    "quantum_mamba_hybrid_lite"
    "quantum_mamba_hybrid_lite"
)

SEEDS=(2024 2025 2026 2027 2028 2024 2025 2026 2027 2028)

# Get model and seed for this array task
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "6-Qubit EEG Lite Experiment"
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
# Using same hyperparameters as existing 6-qubit EEG experiments
# qlcu-layers=2 to match quantum_hydra, quantum_mamba, etc.
python scripts/run_single_model_eeg.py \
    --model-name $MODEL \
    --n-qubits 6 \
    --qlcu-layers 2 \
    --d-model 128 \
    --d-state 16 \
    --n-epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --early-stopping-patience 10 \
    --sample-size 50 \
    --sampling-freq 80 \
    --seed $SEED \
    --output-dir ./results/eeg_results \
    --device cuda

echo ""
echo "============================================================"
echo "Job completed: $MODEL seed=$SEED"
echo "End time: $(date)"
echo "============================================================"

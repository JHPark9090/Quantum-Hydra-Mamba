#!/bin/bash
#SBATCH --job-name=eeg_8q_mamba_lite
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-2
#SBATCH --output=/pscratch/sd/j/junghoon/results/eeg_results_8q_160hz/logs/eeg_8q_mamba_lite_%A_%a.log
#SBATCH --error=/pscratch/sd/j/junghoon/results/eeg_results_8q_160hz/logs/eeg_8q_mamba_lite_%A_%a.log

# Array job for 8-qubit quantum_mamba_lite ONLY at 160 Hz
# Array indices 0-2: seeds 2024-2026
#
# Resubmission after fixing QuantumMamba.py with batched processing
# (removed per-sample loop that caused OOM/hang)
#
# Single GPU configuration (optimal for quantum models)
# Date: November 21, 2025

# Move to project directory
cd /pscratch/sd/j/junghoon

# Activate conda environment
source /pscratch/sd/j/junghoon/.bashrc
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

# Change to project directory
cd /pscratch/sd/j/junghoon/quantum_hydra_mamba

# Seed mapping
SEEDS=(2024 2025 2026)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "8-Qubit quantum_mamba_lite (160 Hz) - Batched Processing Fix"
echo "============================================================"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: quantum_mamba_lite"
echo "Seed: $SEED"
echo "Sampling Frequency: 160 Hz"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L | head -1)"
echo "============================================================"
echo ""

# Run training
python scripts/run_single_model_eeg.py \
    --model-name quantum_mamba_lite \
    --n-qubits 8 \
    --qlcu-layers 2 \
    --d-model 128 \
    --d-state 16 \
    --n-epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --early-stopping-patience 10 \
    --sample-size 50 \
    --sampling-freq 160 \
    --seed $SEED \
    --output-dir ./results/eeg_results_8q_160hz \
    --device cuda

echo ""
echo "============================================================"
echo "Job completed: quantum_mamba_lite seed=$SEED"
echo "End time: $(date)"
echo "============================================================"

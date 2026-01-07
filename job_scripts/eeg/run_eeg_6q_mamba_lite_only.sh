#!/bin/bash
#SBATCH --job-name=eeg_6q_mamba_lite
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-4
#SBATCH --output=/pscratch/sd/j/junghoon/results/eeg_results/logs/eeg_6q_mamba_lite_%A_%a.log
#SBATCH --error=/pscratch/sd/j/junghoon/results/eeg_results/logs/eeg_6q_mamba_lite_%A_%a.log

# Array job for 6-qubit quantum_mamba_lite ONLY
# Array indices 0-4: seeds 2024-2028
#
# Resubmission after fixing QuantumMamba.py with batched processing
# (removed per-sample loop that caused hang)
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
SEEDS=(2024 2025 2026 2027 2028)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "6-Qubit quantum_mamba_lite - Batched Processing Fix"
echo "============================================================"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: quantum_mamba_lite"
echo "Seed: $SEED"
echo "Sampling Frequency: 80 Hz"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L | head -1)"
echo "============================================================"
echo ""

# Run training
python scripts/run_single_model_eeg.py \
    --model-name quantum_mamba_lite \
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
echo "Job completed: quantum_mamba_lite seed=$SEED"
echo "End time: $(date)"
echo "============================================================"

#!/bin/bash
#SBATCH --job-name=eeg_quantum_mamba_10q_s2024
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/pscratch/sd/j/junghoon/quantum_hydra_mamba/results/eeg_results/logs/eeg_quantum_mamba_10q_s2024.log
#SBATCH --error=/pscratch/sd/j/junghoon/quantum_hydra_mamba/results/eeg_results/logs/eeg_quantum_mamba_10q_s2024.log

# Job script for EEG Classification: quantum_mamba with 10 qubits (seed=2024)
# Generated for improved performance testing

# Activate conda environment
source /pscratch/sd/j/junghoon/.bashrc
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

# Change to project directory
cd /pscratch/sd/j/junghoon/quantum_hydra_mamba

# Run training with 10 qubits
python scripts/run_single_model_eeg.py \
    --model-name quantum_mamba \
    --n-qubits 10 \
    --qlcu-layers 2 \
    --d-model 128 \
    --d-state 16 \
    --n-epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --early-stopping-patience 10 \
    --sample-size 50 \
    --sampling-freq 80 \
    --seed 2024 \
    --output-dir ./results/eeg_results \
    --device cuda

echo "Job completed: quantum_mamba 10-qubits seed=2024"

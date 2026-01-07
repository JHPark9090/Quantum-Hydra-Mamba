#!/bin/bash
#SBATCH --job-name=ssm_genomic
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --output=results/genomic_ssm_comparison/logs/ssm_demo_%j.out
#SBATCH --error=results/genomic_ssm_comparison/logs/ssm_demo_%j.err

# ==============================================================================
# SSM Genomic Comparison Experiment
# Dataset: demo_human_or_worm
# Models: QuantumMambaSSM, QuantumHydraSSM, ClassicalMamba, ClassicalHydra
# ==============================================================================

echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="

# Load conda
source ~/.bashrc
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

# Navigate to project directory
cd /pscratch/sd/j/junghoon/quantum_hydra_mamba

# Create output directories
mkdir -p results/genomic_ssm_comparison/logs

# Configuration
DATASET="demo_human_or_worm"
MAX_SAMPLES=2000        # Use more samples for proper comparison
N_EPOCHS=50             # More epochs for convergence
BATCH_SIZE=32
D_MODEL=64
D_STATE=16
N_LAYERS=2
N_QUBITS=4
QLCU_LAYERS=2
SEED=2024

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Max samples: $MAX_SAMPLES"
echo "  Epochs: $N_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  d_model: $D_MODEL, d_state: $D_STATE, n_layers: $N_LAYERS"
echo "  n_qubits: $N_QUBITS, qlcu_layers: $QLCU_LAYERS"
echo "  Seed: $SEED"
echo ""

# Run the comparison experiment
python -u scripts/run_ssm_genomic_comparison.py \
    --dataset $DATASET \
    --max-samples $MAX_SAMPLES \
    --n-epochs $N_EPOCHS \
    --batch-size $BATCH_SIZE \
    --d-model $D_MODEL \
    --d-state $D_STATE \
    --n-layers $N_LAYERS \
    --n-qubits $N_QUBITS \
    --qlcu-layers $QLCU_LAYERS \
    --seed $SEED \
    --device cuda \
    --output-dir results/genomic_ssm_comparison

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "=============================================="

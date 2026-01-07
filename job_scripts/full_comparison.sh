#!/bin/bash
#SBATCH --job-name=full_compare
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --output=results/full_comparison/logs/full_compare_%j.out
#SBATCH --error=results/full_comparison/logs/full_compare_%j.err

# ==============================================================================
# FULL COMPARISON: SSM vs Gated vs Classical Models
#
# Runs all 6 models on 3 small Genomic Benchmark datasets with 3 seeds
# Fair comparison: same n_qubits, qlcu_layers for all quantum models
#
# Estimated time:
# - demo_human_or_worm (200 seq_len): ~1-2 hours per seed
# - demo_coding_vs_intergenomic (200 seq_len): ~1-2 hours per seed
# - drosophila_enhancers_stark (smaller): ~1-2 hours per seed
# Total: ~9-12 hours for all configurations
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
mkdir -p results/full_comparison/logs

# ============================
# Configuration for FAIR comparison
# ============================
# Same settings for ALL quantum models
N_QUBITS=4
QLCU_LAYERS=2
D_MODEL=64
D_STATE=16
N_LAYERS=1
CHUNK_SIZE=16
DROPOUT=0.1

# Training settings
N_EPOCHS=30
BATCH_SIZE=16
LR=0.001
EARLY_STOPPING=15

# Datasets: 3 smallest datasets
DATASETS=(
    "demo_human_or_worm"
    "demo_coding_vs_intergenomic_seqs"
    "drosophila_enhancers_stark"
)

# Seeds for robust results
SEEDS=(2024 2025 2026)

# Max samples per dataset (to keep runtime manageable)
MAX_SAMPLES=500

echo ""
echo "Configuration:"
echo "  n_qubits: $N_QUBITS"
echo "  qlcu_layers: $QLCU_LAYERS"
echo "  d_model: $D_MODEL"
echo "  d_state: $D_STATE"
echo "  n_layers: $N_LAYERS"
echo "  chunk_size: $CHUNK_SIZE"
echo "  n_epochs: $N_EPOCHS"
echo "  batch_size: $BATCH_SIZE"
echo "  max_samples: $MAX_SAMPLES"
echo "  datasets: ${DATASETS[@]}"
echo "  seeds: ${SEEDS[@]}"
echo ""

# Run comparisons (checkpoint system automatically skips completed models)
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "======================================================================"
        echo "Running: $DATASET with seed $SEED"
        echo "======================================================================"
        echo ""

        python -u scripts/run_full_comparison.py \
            --dataset $DATASET \
            --max-samples $MAX_SAMPLES \
            --n-qubits $N_QUBITS \
            --qlcu-layers $QLCU_LAYERS \
            --d-model $D_MODEL \
            --d-state $D_STATE \
            --n-layers $N_LAYERS \
            --chunk-size $CHUNK_SIZE \
            --dropout $DROPOUT \
            --n-epochs $N_EPOCHS \
            --batch-size $BATCH_SIZE \
            --lr $LR \
            --early-stopping $EARLY_STOPPING \
            --seed $SEED \
            --device cuda \
            --output-dir results/full_comparison

        echo ""
        echo "Completed: $DATASET with seed $SEED"
        echo ""
    done
done

echo ""
echo "=============================================="
echo "All comparisons completed!"
echo "End time: $(date)"
echo "=============================================="

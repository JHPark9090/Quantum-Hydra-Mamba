#!/bin/bash
#SBATCH --job-name=glue_cola
#SBATCH --account=m4138_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=/pscratch/sd/j/junghoon/quantum_hydra_mamba/glue/logs/glue_cola_%j.out
#SBATCH --error=/pscratch/sd/j/junghoon/quantum_hydra_mamba/glue/logs/glue_cola_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=junghoon@lbl.gov

# ================================================================================
# GLUE Benchmark: CoLA (Corpus of Linguistic Acceptability)
# Single sentence grammaticality classification
# Metric: Matthews Correlation Coefficient
# ================================================================================

echo "=============================================="
echo "GLUE CoLA Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=============================================="

# Environment setup
cd /pscratch/sd/j/junghoon/quantum_hydra_mamba/glue
source ~/.bashrc
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

# Create directories
mkdir -p logs
mkdir -p glue_results/cola

# Print environment info
echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run experiment
python run_glue.py \
    --task=cola \
    --model=all \
    --embedding-dim=128 \
    --hidden-dim=64 \
    --n-qubits=6 \
    --qlcu-layers=2 \
    --chunk-size=16 \
    --max-length=64 \
    --n-epochs=50 \
    --batch-size=32 \
    --lr=2e-4 \
    --patience=10 \
    --seed=2024 \
    --output-dir=../results/glue/cola \
    --device=cuda

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "=============================================="

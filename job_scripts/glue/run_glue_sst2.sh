#!/bin/bash
#SBATCH --job-name=glue_sst2
#SBATCH --account=m4138_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=/pscratch/sd/j/junghoon/quantum_hydra_mamba/glue/logs/glue_sst2_%j.out
#SBATCH --error=/pscratch/sd/j/junghoon/quantum_hydra_mamba/glue/logs/glue_sst2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=junghoon@lbl.gov

# ================================================================================
# GLUE Benchmark: SST-2 (Stanford Sentiment Treebank)
# Single sentence sentiment classification (positive/negative)
# ================================================================================

echo "=============================================="
echo "GLUE SST-2 Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=============================================="

# Environment setup
cd /pscratch/sd/j/junghoon/quantum_hydra_mamba/glue
source ~/.bashrc
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

# Create log directory
mkdir -p logs

# Optional: Handle GPU memory issues
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Print environment info
echo ""
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# Run experiment
python run_glue.py \
    --task=sst2 \
    --model=all \
    --embedding-dim=128 \
    --hidden-dim=64 \
    --n-qubits=6 \
    --qlcu-layers=2 \
    --chunk-size=16 \
    --max-length=128 \
    --n-epochs=50 \
    --batch-size=32 \
    --lr=2e-4 \
    --patience=10 \
    --seed=2024 \
    --output-dir=../results/glue/sst2 \
    --device=cuda

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "=============================================="

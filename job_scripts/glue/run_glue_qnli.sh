#!/bin/bash
#SBATCH --job-name=glue_qnli
#SBATCH --account=m4138_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=/pscratch/sd/j/junghoon/quantum_hydra_mamba/glue/logs/glue_qnli_%j.out
#SBATCH --error=/pscratch/sd/j/junghoon/quantum_hydra_mamba/glue/logs/glue_qnli_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=junghoon@lbl.gov

# ================================================================================
# GLUE Benchmark: QNLI (Question Natural Language Inference)
# Question-sentence pair entailment (105k samples - MEDIUM-LARGE)
# Metric: Accuracy
# ================================================================================

echo "=============================================="
echo "GLUE QNLI Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=============================================="

cd /pscratch/sd/j/junghoon/quantum_hydra_mamba/glue
source ~/.bashrc
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

mkdir -p logs
mkdir -p glue_results/qnli

echo ""
echo "Python: $(which python)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

python run_glue.py \
    --task=qnli \
    --model=all \
    --embedding-dim=128 \
    --hidden-dim=64 \
    --n-qubits=6 \
    --qlcu-layers=2 \
    --chunk-size=16 \
    --max-length=128 \
    --n-epochs=30 \
    --batch-size=32 \
    --lr=2e-4 \
    --patience=5 \
    --seed=2024 \
    --output-dir=../results/glue/qnli \
    --device=cuda

echo ""
echo "End Time: $(date)"
echo "=============================================="

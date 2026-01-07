#!/bin/bash
#SBATCH --job-name=glue_wnli
#SBATCH --account=m4138_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=/pscratch/sd/j/junghoon/quantum_hydra_mamba/glue/logs/glue_wnli_%j.out
#SBATCH --error=/pscratch/sd/j/junghoon/quantum_hydra_mamba/glue/logs/glue_wnli_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=junghoon@lbl.gov

# ================================================================================
# GLUE Benchmark: WNLI (Winograd Natural Language Inference)
# Sentence pair coreference (634 samples - VERY SMALL, known to be difficult)
# Metric: Accuracy
# Note: This task is notoriously difficult; even BERT often predicts majority class
# ================================================================================

echo "=============================================="
echo "GLUE WNLI Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=============================================="

cd /pscratch/sd/j/junghoon/quantum_hydra_mamba/glue
source ~/.bashrc
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

mkdir -p logs
mkdir -p glue_results/wnli

echo ""
echo "Python: $(which python)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

python run_glue.py \
    --task=wnli \
    --model=all \
    --embedding-dim=128 \
    --hidden-dim=64 \
    --n-qubits=6 \
    --qlcu-layers=2 \
    --chunk-size=16 \
    --max-length=128 \
    --n-epochs=100 \
    --batch-size=8 \
    --lr=1e-4 \
    --patience=20 \
    --seed=2024 \
    --output-dir=../results/glue/wnli \
    --device=cuda

echo ""
echo "End Time: $(date)"
echo "=============================================="

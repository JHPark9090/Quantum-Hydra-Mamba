#!/bin/bash
# Update job scripts to use m4727_g account and submit all failed experiments
# Created: 2025-11-15

echo "================================================================================"
echo "UPDATING JOB SCRIPTS AND SUBMITTING FAILED EXPERIMENTS"
echo "================================================================================"
echo ""

# Create directory for updated scripts
mkdir -p /pscratch/sd/j/junghoon/experiments/updated_job_scripts

total_submitted=0

# ============================================================================
# 1. EEG quantum_hydra experiments (48 hours - long running)
# ============================================================================
echo "1. Creating and submitting EEG quantum_hydra jobs (48h)..."
echo "--------------------------------------------------------------------------------"

for seed in 2024 2025 2026 2027 2028; do
  cat > /pscratch/sd/j/junghoon/experiments/updated_job_scripts/eeg_quantum_hydra_seed${seed}.sh << EOF
#!/bin/bash
#SBATCH --job-name=eeg_qhydra_s${seed}
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./experiments/eeg_results/logs/eeg_quantum_hydra_seed${seed}.log
#SBATCH --error=./experiments/eeg_results/logs/eeg_quantum_hydra_seed${seed}.log

# Job script for EEG Classification: quantum_hydra (seed=${seed})
# Updated with m4727_g account

# Activate conda environment
source activate ./conda-envs/qml_env

# Run training
python experiments/run_single_model_eeg.py \
    --model-name quantum_hydra \
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
    --seed ${seed} \
    --output-dir ./experiments/eeg_results \
    --device cuda

echo "Job completed: quantum_hydra seed=${seed}"
EOF
  chmod +x /pscratch/sd/j/junghoon/experiments/updated_job_scripts/eeg_quantum_hydra_seed${seed}.sh
  echo "  Created: eeg_quantum_hydra_seed${seed}.sh"
done

# ============================================================================
# 2. EEG quantum_hydra_hybrid experiments (48 hours)
# ============================================================================
echo ""
echo "2. Creating and submitting EEG quantum_hydra_hybrid jobs (48h)..."
echo "--------------------------------------------------------------------------------"

for seed in 2024 2025 2026 2027 2028; do
  cat > /pscratch/sd/j/junghoon/experiments/updated_job_scripts/eeg_quantum_hydra_hybrid_seed${seed}.sh << EOF
#!/bin/bash
#SBATCH --job-name=eeg_qhydra_hyb_s${seed}
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./experiments/eeg_results/logs/eeg_quantum_hydra_hybrid_seed${seed}.log
#SBATCH --error=./experiments/eeg_results/logs/eeg_quantum_hydra_hybrid_seed${seed}.log

# Job script for EEG Classification: quantum_hydra_hybrid (seed=${seed})
# Updated with m4727_g account

# Activate conda environment
source activate ./conda-envs/qml_env

# Run training
python experiments/run_single_model_eeg.py \
    --model-name quantum_hydra_hybrid \
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
    --seed ${seed} \
    --output-dir ./experiments/eeg_results \
    --device cuda

echo "Job completed: quantum_hydra_hybrid seed=${seed}"
EOF
  chmod +x /pscratch/sd/j/junghoon/experiments/updated_job_scripts/eeg_quantum_hydra_hybrid_seed${seed}.sh
  echo "  Created: eeg_quantum_hydra_hybrid_seed${seed}.sh"
done

# ============================================================================
# 3. EEG classical_hydra experiments (48 hours - with NaN fix)
# ============================================================================
echo ""
echo "3. Creating and submitting EEG classical_hydra jobs (48h, with NaN fix)..."
echo "--------------------------------------------------------------------------------"

for seed in 2024 2025 2026 2027 2028; do
  cat > /pscratch/sd/j/junghoon/experiments/updated_job_scripts/eeg_classical_hydra_seed${seed}.sh << EOF
#!/bin/bash
#SBATCH --job-name=eeg_chydra_s${seed}
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./experiments/eeg_results/logs/eeg_classical_hydra_seed${seed}.log
#SBATCH --error=./experiments/eeg_results/logs/eeg_classical_hydra_seed${seed}.log

# Job script for EEG Classification: classical_hydra (seed=${seed})
# Updated with m4727_g account and gradient clipping for NaN fix

# Activate conda environment
source activate ./conda-envs/qml_env

# Run training
python experiments/run_single_model_eeg.py \
    --model-name classical_hydra \
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
    --seed ${seed} \
    --output-dir ./experiments/eeg_results \
    --device cuda

echo "Job completed: classical_hydra seed=${seed}"
EOF
  chmod +x /pscratch/sd/j/junghoon/experiments/updated_job_scripts/eeg_classical_hydra_seed${seed}.sh
  echo "  Created: eeg_classical_hydra_seed${seed}.sh"
done

# ============================================================================
# 4. DNA classical_hydra experiments (24 hours)
# ============================================================================
echo ""
echo "4. Creating and submitting DNA classical_hydra jobs (24h)..."
echo "--------------------------------------------------------------------------------"

for seed in 2024 2025 2026 2027 2028; do
  cat > /pscratch/sd/j/junghoon/experiments/updated_job_scripts/dna_classical_hydra_seed${seed}.sh << EOF
#!/bin/bash
#SBATCH --job-name=dna_chydra_s${seed}
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./experiments/dna_results/logs/dna_classical_hydra_seed${seed}.log
#SBATCH --error=./experiments/dna_results/logs/dna_classical_hydra_seed${seed}.log

# Job script for DNA Classification: classical_hydra (seed=${seed})
# Updated with m4727_g account and dimension fix

# Activate conda environment
source activate ./conda-envs/qml_env

# Run training
python experiments/run_single_model_dna.py \
    --model-name classical_hydra \
    --n-qubits 6 \
    --qlcu-layers 2 \
    --d-model 128 \
    --d-state 16 \
    --n-epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --early-stopping-patience 10 \
    --seed ${seed} \
    --output-dir ./experiments/dna_results \
    --device cuda

echo "Job completed: classical_hydra seed=${seed}"
EOF
  chmod +x /pscratch/sd/j/junghoon/experiments/updated_job_scripts/dna_classical_hydra_seed${seed}.sh
  echo "  Created: dna_classical_hydra_seed${seed}.sh"
done

# ============================================================================
# 5. DNA classical_mamba experiments (24 hours)
# ============================================================================
echo ""
echo "5. Creating and submitting DNA classical_mamba jobs (24h)..."
echo "--------------------------------------------------------------------------------"

for seed in 2024 2025 2026; do
  cat > /pscratch/sd/j/junghoon/experiments/updated_job_scripts/dna_classical_mamba_seed${seed}.sh << EOF
#!/bin/bash
#SBATCH --job-name=dna_cmamba_s${seed}
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./experiments/dna_results/logs/dna_classical_mamba_seed${seed}.log
#SBATCH --error=./experiments/dna_results/logs/dna_classical_mamba_seed${seed}.log

# Job script for DNA Classification: classical_mamba (seed=${seed})
# Updated with m4727_g account and dimension fix

# Activate conda environment
source activate ./conda-envs/qml_env

# Run training
python experiments/run_single_model_dna.py \
    --model-name classical_mamba \
    --n-qubits 6 \
    --qlcu-layers 2 \
    --d-model 128 \
    --d-state 16 \
    --n-epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --early-stopping-patience 10 \
    --seed ${seed} \
    --output-dir ./experiments/dna_results \
    --device cuda

echo "Job completed: classical_mamba seed=${seed}"
EOF
  chmod +x /pscratch/sd/j/junghoon/experiments/updated_job_scripts/dna_classical_mamba_seed${seed}.sh
  echo "  Created: dna_classical_mamba_seed${seed}.sh"
done

# ============================================================================
# 6. MNIST classical_hydra experiments (12 hours)
# ============================================================================
echo ""
echo "6. Creating and submitting MNIST classical_hydra jobs (12h)..."
echo "--------------------------------------------------------------------------------"

for seed in 2024 2025 2026 2027 2028; do
  cat > /pscratch/sd/j/junghoon/experiments/updated_job_scripts/mnist_classical_hydra_seed${seed}.sh << EOF
#!/bin/bash
#SBATCH --job-name=mnist_chydra_s${seed}
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./experiments/mnist_results/logs/mnist_classical_hydra_seed${seed}.log
#SBATCH --error=./experiments/mnist_results/logs/mnist_classical_hydra_seed${seed}.log

# Job script for MNIST Classification: classical_hydra (seed=${seed})
# Updated with m4727_g account and dimension fix

# Activate conda environment
source activate ./conda-envs/qml_env

# Run training
python experiments/run_single_model_mnist.py \
    --model-name classical_hydra \
    --n-qubits 6 \
    --qlcu-layers 2 \
    --d-model 128 \
    --d-state 16 \
    --n-epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --n-train 500 \
    --n-valtest 250 \
    --early-stopping-patience 10 \
    --seed ${seed} \
    --output-dir ./experiments/mnist_results \
    --device cuda

echo "Job completed: classical_hydra seed=${seed}"
EOF
  chmod +x /pscratch/sd/j/junghoon/experiments/updated_job_scripts/mnist_classical_hydra_seed${seed}.sh
  echo "  Created: mnist_classical_hydra_seed${seed}.sh"
done

# ============================================================================
# 7. MNIST classical_mamba experiments (12 hours)
# ============================================================================
echo ""
echo "7. Creating and submitting MNIST classical_mamba jobs (12h)..."
echo "--------------------------------------------------------------------------------"

for seed in 2024 2025 2026 2027 2028; do
  cat > /pscratch/sd/j/junghoon/experiments/updated_job_scripts/mnist_classical_mamba_seed${seed}.sh << EOF
#!/bin/bash
#SBATCH --job-name=mnist_cmamba_s${seed}
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./experiments/mnist_results/logs/mnist_classical_mamba_seed${seed}.log
#SBATCH --error=./experiments/mnist_results/logs/mnist_classical_mamba_seed${seed}.log

# Job script for MNIST Classification: classical_mamba (seed=${seed})
# Updated with m4727_g account and dimension fix

# Activate conda environment
source activate ./conda-envs/qml_env

# Run training
python experiments/run_single_model_mnist.py \
    --model-name classical_mamba \
    --n-qubits 6 \
    --qlcu-layers 2 \
    --d-model 128 \
    --d-state 16 \
    --n-epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --n-train 500 \
    --n-valtest 250 \
    --early-stopping-patience 10 \
    --seed ${seed} \
    --output-dir ./experiments/mnist_results \
    --device cuda

echo "Job completed: classical_mamba seed=${seed}"
EOF
  chmod +x /pscratch/sd/j/junghoon/experiments/updated_job_scripts/mnist_classical_mamba_seed${seed}.sh
  echo "  Created: mnist_classical_mamba_seed${seed}.sh"
done

echo ""
echo "================================================================================"
echo "All job scripts created successfully!"
echo "Total scripts created: 33"
echo "================================================================================"
echo ""
echo "Job breakdown:"
echo "  - EEG quantum_hydra:         5 jobs (48h each)"
echo "  - EEG quantum_hydra_hybrid:  5 jobs (48h each)"
echo "  - EEG classical_hydra:       5 jobs (48h each)"
echo "  - DNA classical_hydra:       5 jobs (24h each)"
echo "  - DNA classical_mamba:       3 jobs (24h each)"
echo "  - MNIST classical_hydra:     5 jobs (12h each)"
echo "  - MNIST classical_mamba:     5 jobs (12h each)"
echo ""
echo "Ready to submit? (y/n)"
read -p "> " response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Submitting all jobs..."
    echo "================================================================================"

    cd /pscratch/sd/j/junghoon

    # Submit all jobs
    for script in /pscratch/sd/j/junghoon/experiments/updated_job_scripts/*.sh; do
        job_id=$(sbatch "$script" 2>&1 | grep -oP 'Submitted batch job \K\d+')
        if [ -n "$job_id" ]; then
            echo "✓ Submitted: $(basename $script) (Job ID: $job_id)"
            ((total_submitted++))
        else
            echo "✗ Failed: $(basename $script)"
        fi
    done

    echo ""
    echo "================================================================================"
    echo "Submission complete!"
    echo "Total jobs submitted: $total_submitted/33"
    echo "================================================================================"
    echo ""
    echo "Monitor jobs with:"
    echo "  squeue -u \$USER"
    echo ""
    echo "Check logs in:"
    echo "  experiments/eeg_results/logs/"
    echo "  experiments/dna_results/logs/"
    echo "  experiments/mnist_results/logs/"
    echo ""
else
    echo ""
    echo "Job submission cancelled."
    echo "To submit later, run individual scripts from:"
    echo "  /pscratch/sd/j/junghoon/experiments/updated_job_scripts/"
    echo ""
fi

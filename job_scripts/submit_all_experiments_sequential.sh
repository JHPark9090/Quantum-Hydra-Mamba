#!/bin/bash
# Sequential Submission Script for All 90 Tier 1 Experiments
# Runs jobs one at a time to avoid GPU memory conflicts

echo "================================================================================"
echo "SEQUENTIAL JOB SUBMISSION FOR TIER 1 EXPERIMENTS"
echo "================================================================================"
echo "Total jobs: 90 (30 EEG + 30 MNIST + 30 DNA)"
echo "Strategy: Run one job at a time to completion"
echo "================================================================================"
echo ""

# Counter
job_count=0
total_jobs=90

# Array of all job scripts
declare -a ALL_JOBS=(
    # EEG jobs (30)
    "job_scripts/eeg/eeg_quantum_hydra_seed2024.sh"
    "job_scripts/eeg/eeg_quantum_hydra_seed2025.sh"
    "job_scripts/eeg/eeg_quantum_hydra_seed2026.sh"
    "job_scripts/eeg/eeg_quantum_hydra_seed2027.sh"
    "job_scripts/eeg/eeg_quantum_hydra_seed2028.sh"
    "job_scripts/eeg/eeg_quantum_hydra_hybrid_seed2024.sh"
    "job_scripts/eeg/eeg_quantum_hydra_hybrid_seed2025.sh"
    "job_scripts/eeg/eeg_quantum_hydra_hybrid_seed2026.sh"
    "job_scripts/eeg/eeg_quantum_hydra_hybrid_seed2027.sh"
    "job_scripts/eeg/eeg_quantum_hydra_hybrid_seed2028.sh"
    "job_scripts/eeg/eeg_quantum_mamba_seed2024.sh"
    "job_scripts/eeg/eeg_quantum_mamba_seed2025.sh"
    "job_scripts/eeg/eeg_quantum_mamba_seed2026.sh"
    "job_scripts/eeg/eeg_quantum_mamba_seed2027.sh"
    "job_scripts/eeg/eeg_quantum_mamba_seed2028.sh"
    "job_scripts/eeg/eeg_quantum_mamba_hybrid_seed2024.sh"
    "job_scripts/eeg/eeg_quantum_mamba_hybrid_seed2025.sh"
    "job_scripts/eeg/eeg_quantum_mamba_hybrid_seed2026.sh"
    "job_scripts/eeg/eeg_quantum_mamba_hybrid_seed2027.sh"
    "job_scripts/eeg/eeg_quantum_mamba_hybrid_seed2028.sh"
    "job_scripts/eeg/eeg_classical_hydra_seed2024.sh"
    "job_scripts/eeg/eeg_classical_hydra_seed2025.sh"
    "job_scripts/eeg/eeg_classical_hydra_seed2026.sh"
    "job_scripts/eeg/eeg_classical_hydra_seed2027.sh"
    "job_scripts/eeg/eeg_classical_hydra_seed2028.sh"
    "job_scripts/eeg/eeg_classical_mamba_seed2024.sh"
    "job_scripts/eeg/eeg_classical_mamba_seed2025.sh"
    "job_scripts/eeg/eeg_classical_mamba_seed2026.sh"
    "job_scripts/eeg/eeg_classical_mamba_seed2027.sh"
    "job_scripts/eeg/eeg_classical_mamba_seed2028.sh"

    # MNIST jobs (30)
    "job_scripts/mnist_quantum_hydra_seed2024.sh"
    "job_scripts/mnist_quantum_hydra_seed2025.sh"
    "job_scripts/mnist_quantum_hydra_seed2026.sh"
    "job_scripts/mnist_quantum_hydra_seed2027.sh"
    "job_scripts/mnist_quantum_hydra_seed2028.sh"
    "job_scripts/mnist_quantum_hydra_hybrid_seed2024.sh"
    "job_scripts/mnist_quantum_hydra_hybrid_seed2025.sh"
    "job_scripts/mnist_quantum_hydra_hybrid_seed2026.sh"
    "job_scripts/mnist_quantum_hydra_hybrid_seed2027.sh"
    "job_scripts/mnist_quantum_hydra_hybrid_seed2028.sh"
    "job_scripts/mnist_quantum_mamba_seed2024.sh"
    "job_scripts/mnist_quantum_mamba_seed2025.sh"
    "job_scripts/mnist_quantum_mamba_seed2026.sh"
    "job_scripts/mnist_quantum_mamba_seed2027.sh"
    "job_scripts/mnist_quantum_mamba_seed2028.sh"
    "job_scripts/mnist_quantum_mamba_hybrid_seed2024.sh"
    "job_scripts/mnist_quantum_mamba_hybrid_seed2025.sh"
    "job_scripts/mnist_quantum_mamba_hybrid_seed2026.sh"
    "job_scripts/mnist_quantum_mamba_hybrid_seed2027.sh"
    "job_scripts/mnist_quantum_mamba_hybrid_seed2028.sh"
    "job_scripts/mnist_classical_hydra_seed2024.sh"
    "job_scripts/mnist_classical_hydra_seed2025.sh"
    "job_scripts/mnist_classical_hydra_seed2026.sh"
    "job_scripts/mnist_classical_hydra_seed2027.sh"
    "job_scripts/mnist_classical_hydra_seed2028.sh"
    "job_scripts/mnist_classical_mamba_seed2024.sh"
    "job_scripts/mnist_classical_mamba_seed2025.sh"
    "job_scripts/mnist_classical_mamba_seed2026.sh"
    "job_scripts/mnist_classical_mamba_seed2027.sh"
    "job_scripts/mnist_classical_mamba_seed2028.sh"

    # DNA jobs (30)
    "job_scripts/dna/dna_quantum_hydra_seed2024.sh"
    "job_scripts/dna/dna_quantum_hydra_seed2025.sh"
    "job_scripts/dna/dna_quantum_hydra_seed2026.sh"
    "job_scripts/dna/dna_quantum_hydra_seed2027.sh"
    "job_scripts/dna/dna_quantum_hydra_seed2028.sh"
    "job_scripts/dna/dna_quantum_hydra_hybrid_seed2024.sh"
    "job_scripts/dna/dna_quantum_hydra_hybrid_seed2025.sh"
    "job_scripts/dna/dna_quantum_hydra_hybrid_seed2026.sh"
    "job_scripts/dna/dna_quantum_hydra_hybrid_seed2027.sh"
    "job_scripts/dna/dna_quantum_hydra_hybrid_seed2028.sh"
    "job_scripts/dna/dna_quantum_mamba_seed2024.sh"
    "job_scripts/dna/dna_quantum_mamba_seed2025.sh"
    "job_scripts/dna/dna_quantum_mamba_seed2026.sh"
    "job_scripts/dna/dna_quantum_mamba_seed2027.sh"
    "job_scripts/dna/dna_quantum_mamba_seed2028.sh"
    "job_scripts/dna/dna_quantum_mamba_hybrid_seed2024.sh"
    "job_scripts/dna/dna_quantum_mamba_hybrid_seed2025.sh"
    "job_scripts/dna/dna_quantum_mamba_hybrid_seed2026.sh"
    "job_scripts/dna/dna_quantum_mamba_hybrid_seed2027.sh"
    "job_scripts/dna/dna_quantum_mamba_hybrid_seed2028.sh"
    "job_scripts/dna/dna_classical_hydra_seed2024.sh"
    "job_scripts/dna/dna_classical_hydra_seed2025.sh"
    "job_scripts/dna/dna_classical_hydra_seed2026.sh"
    "job_scripts/dna/dna_classical_hydra_seed2027.sh"
    "job_scripts/dna/dna_classical_hydra_seed2028.sh"
    "job_scripts/dna/dna_classical_mamba_seed2024.sh"
    "job_scripts/dna/dna_classical_mamba_seed2026.sh"
    "job_scripts/dna/dna_classical_mamba_seed2027.sh"
    "job_scripts/dna/dna_classical_mamba_seed2028.sh"
)

# Run each job sequentially
for job_script in "${ALL_JOBS[@]}"; do
    ((job_count++))

    job_name=$(basename "$job_script" .sh)

    echo "--------------------------------------------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $job_count/$total_jobs: $job_name"
    echo "--------------------------------------------------------------------------------"

    # Run job and wait for completion
    bash "$job_script"
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ Job completed successfully: $job_name"
    else
        echo "✗ Job failed with exit code $exit_code: $job_name"
    fi

    echo ""

    # Small pause between jobs
    sleep 2
done

echo "================================================================================"
echo "ALL JOBS COMPLETED!"
echo "================================================================================"
echo "Total jobs run: $job_count"
echo "Check results in:"
echo "  - results/eeg_results/"
echo "  - results/mnist_results/"
echo "  - results/dna_results/"
echo "================================================================================"

#!/bin/bash
# Batched Submission Script for All 90 Tier 1 Experiments
# Runs jobs in batches of 5 to avoid GPU memory conflicts

BATCH_SIZE=5
WAIT_TIME=10  # seconds between batches

echo "================================================================================"
echo "BATCHED JOB SUBMISSION FOR TIER 1 EXPERIMENTS"
echo "================================================================================"
echo "Total jobs: 90 (30 EEG + 30 MNIST + 30 DNA)"
echo "Strategy: Run $BATCH_SIZE jobs at a time with ${WAIT_TIME}s pauses"
echo "================================================================================"
echo ""

# Function to wait for background jobs to complete
wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $BATCH_SIZE ]; do
        sleep 5
    done
}

# Counter
job_count=0

# EEG Jobs (30)
echo "Starting EEG jobs..."
for model in quantum_hydra quantum_hydra_hybrid quantum_mamba quantum_mamba_hybrid classical_hydra classical_mamba; do
    for seed in 2024 2025 2026 2027 2028; do
        ((job_count++))
        job_script="job_scripts/eeg/eeg_${model}_seed${seed}.sh"
        log_file="results/eeg_results/logs/eeg_${model}_seed${seed}.sh.log"

        echo "[$job_count/90] Submitting: eeg_${model}_seed${seed}"
        bash "$job_script" > "$log_file" 2>&1 &

        # Wait if we've reached batch size
        if [ $((job_count % BATCH_SIZE)) -eq 0 ]; then
            echo "  Waiting for batch to complete..."
            wait_for_jobs
            echo "  Batch completed. Continuing..."
            sleep $WAIT_TIME
        fi
    done
done

# Wait for remaining EEG jobs
wait

echo ""
echo "EEG jobs completed. Moving to MNIST..."
echo ""

# MNIST Jobs (30)
for model in quantum_hydra quantum_hydra_hybrid quantum_mamba quantum_mamba_hybrid classical_hydra classical_mamba; do
    for seed in 2024 2025 2026 2027 2028; do
        ((job_count++))
        job_script="job_scripts/mnist_${model}_seed${seed}.sh"
        log_file="results/mnist_results/logs/mnist_${model}_seed${seed}.sh.log"

        echo "[$job_count/90] Submitting: mnist_${model}_seed${seed}"
        bash "$job_script" > "$log_file" 2>&1 &

        # Wait if we've reached batch size
        if [ $((job_count % BATCH_SIZE)) -eq 0 ]; then
            echo "  Waiting for batch to complete..."
            wait_for_jobs
            echo "  Batch completed. Continuing..."
            sleep $WAIT_TIME
        fi
    done
done

# Wait for remaining MNIST jobs
wait

echo ""
echo "MNIST jobs completed. Moving to DNA..."
echo ""

# DNA Jobs (30)
for model in quantum_hydra quantum_hydra_hybrid quantum_mamba quantum_mamba_hybrid classical_hydra classical_mamba; do
    for seed in 2024 2025 2026 2027 2028; do
        ((job_count++))
        job_script="job_scripts/dna/dna_${model}_seed${seed}.sh"
        log_file="results/dna_results/logs/dna_${model}_seed${seed}.sh.log"

        echo "[$job_count/90] Submitting: dna_${model}_seed${seed}"
        bash "$job_script" > "$log_file" 2>&1 &

        # Wait if we've reached batch size
        if [ $((job_count % BATCH_SIZE)) -eq 0 ]; then
            echo "  Waiting for batch to complete..."
            wait_for_jobs
            echo "  Batch completed. Continuing..."
            sleep $WAIT_TIME
        fi
    done
done

# Wait for all remaining jobs
wait

echo ""
echo "================================================================================"
echo "ALL 90 JOBS COMPLETED!"
echo "================================================================================"
echo "Results saved in:"
echo "  - results/eeg_results/"
echo "  - results/mnist_results/"
echo "  - results/dna_results/"
echo ""
echo "Next steps:"
echo "  1. Verify all results files:"
echo "     ls results/*/[!logs]*_results.json | wc -l  # Should be 90"
echo ""
echo "  2. Aggregate results:"
echo "     python scripts/aggregate_eeg_results.py"
echo "     python scripts/aggregate_mnist_results.py"
echo "     python scripts/aggregate_dna_results.py"
echo "================================================================================"

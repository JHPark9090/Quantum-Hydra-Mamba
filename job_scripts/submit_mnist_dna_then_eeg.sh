#!/bin/bash
# Phase 1: Run MNIST & DNA (60 jobs)
# Phase 2: Run EEG with updated config (30 jobs)
# Run in batches of 5 to avoid GPU memory issues

BATCH_SIZE=5
WAIT_TIME=10

echo "================================================================================"
echo "TIER 1 EXPERIMENTS - PHASED SUBMISSION"
echo "================================================================================"
echo "Phase 1: MNIST (30) + DNA (30) = 60 jobs"
echo "Phase 2: EEG with 50 subjects + 80 Hz (30 jobs)"
echo "Batch size: $BATCH_SIZE jobs at a time"
echo "================================================================================"
echo ""

# Function to wait for background jobs to complete
wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $BATCH_SIZE ]; do
        sleep 5
    done
}

job_count=0

# ==============================================================================
# PHASE 1: MNIST EXPERIMENTS (30 jobs)
# ==============================================================================

echo "================================================================================"
echo "PHASE 1A: MNIST CLASSIFICATION (30 jobs)"
echo "================================================================================"
echo ""

for model in quantum_hydra quantum_hydra_hybrid quantum_mamba quantum_mamba_hybrid classical_hydra classical_mamba; do
    for seed in 2024 2025 2026 2027 2028; do
        ((job_count++))
        job_script="job_scripts/mnist_${model}_seed${seed}.sh"
        log_file="results/mnist_results/logs/mnist_${model}_seed${seed}.sh.log"

        echo "[$(date '+%H:%M:%S')] [$job_count/60] Submitting: mnist_${model}_seed${seed}"
        bash "$job_script" > "$log_file" 2>&1 &

        # Wait if we've reached batch size
        if [ $((job_count % BATCH_SIZE)) -eq 0 ]; then
            echo "  → Waiting for batch to complete..."
            wait_for_jobs
            echo "  ✓ Batch completed. Continuing..."
            sleep $WAIT_TIME
        fi
    done
done

# Wait for remaining MNIST jobs
echo ""
echo "Waiting for all MNIST jobs to complete..."
wait
echo "✓ All 30 MNIST jobs completed!"
echo ""

# ==============================================================================
# PHASE 1B: DNA EXPERIMENTS (30 jobs)
# ==============================================================================

echo "================================================================================"
echo "PHASE 1B: DNA SEQUENCE CLASSIFICATION (30 jobs)"
echo "================================================================================"
echo ""

for model in quantum_hydra quantum_hydra_hybrid quantum_mamba quantum_mamba_hybrid classical_hydra classical_mamba; do
    for seed in 2024 2025 2026 2027 2028; do
        ((job_count++))
        job_script="job_scripts/dna/dna_${model}_seed${seed}.sh"
        log_file="results/dna_results/logs/dna_${model}_seed${seed}.sh.log"

        echo "[$(date '+%H:%M:%S')] [$job_count/60] Submitting: dna_${model}_seed${seed}"
        bash "$job_script" > "$log_file" 2>&1 &

        # Wait if we've reached batch size
        if [ $((job_count % BATCH_SIZE)) -eq 0 ]; then
            echo "  → Waiting for batch to complete..."
            wait_for_jobs
            echo "  ✓ Batch completed. Continuing..."
            sleep $WAIT_TIME
        fi
    done
done

# Wait for remaining DNA jobs
echo ""
echo "Waiting for all DNA jobs to complete..."
wait
echo "✓ All 30 DNA jobs completed!"
echo ""

# ==============================================================================
# PHASE 2: EEG EXPERIMENTS (30 jobs) - UPDATED CONFIG
# ==============================================================================

echo "================================================================================"
echo "PHASE 2: EEG MOTOR IMAGERY (30 jobs) - 50 subjects, 80 Hz"
echo "================================================================================"
echo ""

job_count=0

for model in quantum_hydra quantum_hydra_hybrid quantum_mamba quantum_mamba_hybrid classical_hydra classical_mamba; do
    for seed in 2024 2025 2026 2027 2028; do
        ((job_count++))
        job_script="job_scripts/eeg/eeg_${model}_seed${seed}.sh"
        log_file="results/eeg_results/logs/eeg_${model}_seed${seed}.sh.log"

        echo "[$(date '+%H:%M:%S')] [$job_count/30] Submitting: eeg_${model}_seed${seed}"
        bash "$job_script" > "$log_file" 2>&1 &

        # Wait if we've reached batch size
        if [ $((job_count % BATCH_SIZE)) -eq 0 ]; then
            echo "  → Waiting for batch to complete..."
            wait_for_jobs
            echo "  ✓ Batch completed. Continuing..."
            sleep $WAIT_TIME
        fi
    done
done

# Wait for all remaining jobs
echo ""
echo "Waiting for all EEG jobs to complete..."
wait
echo "✓ All 30 EEG jobs completed!"
echo ""

# ==============================================================================
# COMPLETION SUMMARY
# ==============================================================================

echo "================================================================================"
echo "ALL 90 EXPERIMENTS COMPLETED!"
echo "================================================================================"
echo ""
echo "Phase 1 (MNIST + DNA): 60 jobs ✓"
echo "Phase 2 (EEG):         30 jobs ✓"
echo "Total:                 90 jobs ✓"
echo ""
echo "Results saved in:"
echo "  - results/mnist_results/"
echo "  - results/dna_results/"
echo "  - results/eeg_results/"
echo ""
echo "Verify completion:"
echo "  Total results: \$(ls results/*/[!logs]*_results.json | wc -l) (should be 90)"
echo ""
echo "Next steps:"
echo "  1. python scripts/aggregate_mnist_results.py"
echo "  2. python scripts/aggregate_dna_results.py"
echo "  3. python scripts/aggregate_eeg_results.py"
echo "================================================================================"

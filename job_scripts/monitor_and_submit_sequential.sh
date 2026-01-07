#!/bin/bash
# Automated sequential submission script
# Monitors MNIST completion, then submits DNA, then EEG

echo "=========================================="
echo "Sequential Experiment Submission Monitor"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Phase 1: Wait for MNIST to complete
echo "Phase 1: Monitoring MNIST jobs (30 total)..."
while true; do
    mnist_complete=$(ls results/mnist_results/*.json 2>/dev/null | wc -l)
    mnist_running=$(squeue -u $USER -t RUNNING,PENDING -o "%.30j" | grep mnist | wc -l)

    echo "[$(date +%H:%M:%S)] MNIST: $mnist_complete/30 complete, $mnist_running running"

    if [ "$mnist_complete" -eq 30 ] && [ "$mnist_running" -eq 0 ]; then
        echo ""
        echo "✅ MNIST COMPLETE! All 30 jobs finished."
        echo "Completed at: $(date)"
        break
    fi

    sleep 60  # Check every minute
done

echo ""
echo "Waiting 30 seconds before starting DNA jobs..."
sleep 30

# Phase 2: Submit DNA jobs
echo ""
echo "=========================================="
echo "Phase 2: Submitting DNA jobs (30 total)..."
echo "=========================================="
bash job_scripts/dna/submit_all_dna_jobs.sh

echo ""
echo "Waiting for DNA jobs to complete..."
while true; do
    dna_complete=$(ls results/dna_results/*.json 2>/dev/null | wc -l)
    dna_running=$(squeue -u $USER -t RUNNING,PENDING -o "%.30j" | grep dna | wc -l)

    echo "[$(date +%H:%M:%S)] DNA: $dna_complete/30 complete, $dna_running running"

    if [ "$dna_complete" -eq 30 ] && [ "$dna_running" -eq 0 ]; then
        echo ""
        echo "✅ DNA COMPLETE! All 30 jobs finished."
        echo "Completed at: $(date)"
        break
    fi

    sleep 60  # Check every minute
done

echo ""
echo "Waiting 30 seconds before starting EEG jobs..."
sleep 30

# Phase 3: Submit EEG jobs
echo ""
echo "=========================================="
echo "Phase 3: Submitting EEG jobs (30 total)..."
echo "=========================================="
bash job_scripts/eeg/submit_all_eeg_jobs.sh

echo ""
echo "Waiting for EEG jobs to complete..."
while true; do
    eeg_complete=$(ls results/eeg_results/*.json 2>/dev/null | wc -l)
    eeg_running=$(squeue -u $USER -t RUNNING,PENDING -o "%.30j" | grep eeg | wc -l)

    echo "[$(date +%H:%M:%S)] EEG: $eeg_complete/30 complete, $eeg_running running"

    if [ "$eeg_complete" -eq 30 ] && [ "$eeg_running" -eq 0 ]; then
        echo ""
        echo "✅ EEG COMPLETE! All 30 jobs finished."
        echo "Completed at: $(date)"
        break
    fi

    sleep 60  # Check every minute
done

# All done!
echo ""
echo "=========================================="
echo "✅ ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Final Results:"
echo "  MNIST: $(ls results/mnist_results/*.json 2>/dev/null | wc -l)/30"
echo "  DNA: $(ls results/dna_results/*.json 2>/dev/null | wc -l)/30"
echo "  EEG: $(ls results/eeg_results/*.json 2>/dev/null | wc -l)/30"
echo ""
echo "Next steps:"
echo "  1. python scripts/aggregate_mnist_results.py"
echo "  2. python scripts/aggregate_dna_results.py"
echo "  3. python scripts/aggregate_eeg_results.py"

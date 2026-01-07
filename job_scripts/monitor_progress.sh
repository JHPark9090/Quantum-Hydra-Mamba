#!/bin/bash
# Monitor experiment progress

echo "=== Experiment Progress Monitor ==="
echo "Started: $(date)"
echo ""

while true; do
    clear
    echo "=== Experiment Progress - $(date '+%H:%M:%S') ==="
    echo ""

    echo "Running Processes:"
    mnist_procs=$(ps aux | grep "run_single_model_mnist" | grep -v grep | wc -l)
    dna_procs=$(ps aux | grep "run_single_model_dna" | grep -v grep | wc -l)
    eeg_procs=$(ps aux | grep "run_single_model_eeg" | grep -v grep | wc -l)

    echo "  MNIST: $mnist_procs running"
    echo "  DNA:   $dna_procs running"
    echo "  EEG:   $eeg_procs running"
    echo ""

    echo "Completed Results:"
    mnist_done=$(ls results/mnist_results/*_results.json 2>/dev/null | wc -l)
    dna_done=$(ls results/dna_results/*_results.json 2>/dev/null | wc -l)
    eeg_done=$(ls results/eeg_results/*_results.json 2>/dev/null | wc -l)
    total_done=$((mnist_done + dna_done + eeg_done))

    echo "  MNIST: $mnist_done / 30"
    echo "  DNA:   $dna_done / 30"
    echo "  EEG:   $eeg_done / 30"
    echo "  Total: $total_done / 90"
    echo ""

    if [ $total_done -gt 0 ]; then
        echo "Recent completions:"
        ls -lt results/*/mnist_*_results.json results/*/dna_*_results.json results/*/eeg_*_results.json 2>/dev/null | head -5
        echo ""
    fi

    echo "Phased submission status:"
    tail -3 logs/phased_submission.log 2>/dev/null || echo "  (log not available)"
    echo ""

    echo "Press Ctrl+C to stop monitoring..."
    sleep 30
done

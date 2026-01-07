#!/bin/bash
# Monitoring script for submitted quantum experiments
# Usage: bash monitor_experiments.sh

echo "================================================================================"
echo "QUANTUM HYDRA/MAMBA EXPERIMENT MONITORING"
echo "================================================================================"
echo ""
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ============================================================================
# 1. Check SLURM Queue Status
# ============================================================================
echo "1. SLURM QUEUE STATUS"
echo "--------------------------------------------------------------------------------"
total_jobs=$(squeue -u junghoon | grep -c "junghoon")
pending_jobs=$(squeue -u junghoon | grep -c "PD")
running_jobs=$(squeue -u junghoon | grep -c " R ")

echo "Total jobs in queue: $total_jobs"
echo "  - Pending (PD): $pending_jobs"
echo "  - Running (R):  $running_jobs"
echo ""

if [ $running_jobs -gt 0 ]; then
    echo "Running jobs:"
    squeue -u junghoon | grep " R " | awk '{print "  - " $3 " (Job ID: " $1 ")"}'
    echo ""
fi

# ============================================================================
# 2. Check Submitted Experiment Status
# ============================================================================
echo "2. SUBMITTED EXPERIMENT STATUS (Job IDs: 45232061-45232094)"
echo "--------------------------------------------------------------------------------"

# Count by status
dna_pending=0
dna_running=0
mnist_pending=0
mnist_running=0
eeg_pending=0
eeg_running=0

for job_id in {45232061..45232094}; do
    status=$(squeue -j $job_id 2>/dev/null | tail -n +2 | awk '{print $5}')
    name=$(squeue -j $job_id 2>/dev/null | tail -n +2 | awk '{print $3}')

    if [ -n "$status" ]; then
        # DNA experiments
        if [[ $name == dna_* ]]; then
            if [ "$status" == "PD" ]; then
                ((dna_pending++))
            elif [ "$status" == "R" ]; then
                ((dna_running++))
            fi
        fi

        # MNIST experiments
        if [[ $name == mnist_* ]]; then
            if [ "$status" == "PD" ]; then
                ((mnist_pending++))
            elif [ "$status" == "R" ]; then
                ((mnist_running++))
            fi
        fi

        # EEG experiments
        if [[ $name == eeg_* ]]; then
            if [ "$status" == "PD" ]; then
                ((eeg_pending++))
            elif [ "$status" == "R" ]; then
                ((eeg_running++))
            fi
        fi
    fi
done

echo "DNA experiments (8 total):"
echo "  - Pending: $dna_pending"
echo "  - Running: $dna_running"
echo "  - Completed: $((8 - dna_pending - dna_running))"
echo ""

echo "MNIST experiments (10 total):"
echo "  - Pending: $mnist_pending"
echo "  - Running: $mnist_running"
echo "  - Completed: $((10 - mnist_pending - mnist_running))"
echo ""

echo "EEG experiments (15 total):"
echo "  - Pending: $eeg_pending"
echo "  - Running: $eeg_running"
echo "  - Completed: $((15 - eeg_pending - eeg_running))"
echo ""

# ============================================================================
# 3. Check Completed Experiments
# ============================================================================
echo "3. COMPLETED EXPERIMENTS CHECK"
echo "--------------------------------------------------------------------------------"

# Function to count successful experiments in a directory
count_successes() {
    local log_dir=$1
    local pattern=$2
    local success_count=$(grep -l "TRAINING COMPLETE" ${log_dir}/${pattern}*.log 2>/dev/null | wc -l)
    echo $success_count
}

# Function to count failures
count_failures() {
    local log_dir=$1
    local pattern=$2
    local error_count=$(grep -l "Traceback\|Error:" ${log_dir}/${pattern}*.log 2>/dev/null | wc -l)
    echo $error_count
}

# DNA experiments
dna_success=$(count_successes "results/dna_results/logs" "dna_classical")
dna_errors=$(count_failures "results/dna_results/logs" "dna_classical")
echo "DNA (classical_hydra + classical_mamba):"
echo "  ✅ Successful: $dna_success/8"
echo "  ❌ Errors: $dna_errors/8"
echo ""

# MNIST experiments
mnist_success=$(count_successes "results/mnist_results/logs" "mnist_classical")
mnist_errors=$(count_failures "results/mnist_results/logs" "mnist_classical")
echo "MNIST (classical_hydra + classical_mamba):"
echo "  ✅ Successful: $mnist_success/10"
echo "  ❌ Errors: $mnist_errors/10"
echo ""

# EEG experiments
eeg_success=$(count_successes "results/eeg_results/logs" "eeg_")
eeg_errors=$(count_failures "results/eeg_results/logs" "eeg_")
echo "EEG (quantum_hydra + quantum_hydra_hybrid + classical_hydra):"
echo "  ✅ Successful: $eeg_success/15"
echo "  ❌ Errors: $eeg_errors/15"
echo ""

# ============================================================================
# 4. Check for Specific Issues
# ============================================================================
echo "4. ERROR CHECKING"
echo "--------------------------------------------------------------------------------"

# Check for IndexError (should NOT appear with fixes)
index_errors=$(grep -r "IndexError" results/*/logs/*.log 2>/dev/null | wc -l)
if [ $index_errors -gt 0 ]; then
    echo "⚠️  WARNING: Found $index_errors IndexError occurrences (check dimension fixes)"
else
    echo "✅ No IndexError found (dimension fixes working)"
fi

# Check for NaN issues (should NOT appear with gradient clipping)
nan_errors=$(grep -r "Train Loss: nan" results/*/logs/*.log 2>/dev/null | wc -l)
if [ $nan_errors -gt 0 ]; then
    echo "⚠️  WARNING: Found $nan_errors NaN loss occurrences (check gradient clipping)"
else
    echo "✅ No NaN loss found (gradient clipping working)"
fi

# Check for CUDA errors
cuda_errors=$(grep -r "CUDA error\|out of memory" results/*/logs/*.log 2>/dev/null | wc -l)
if [ $cuda_errors -gt 0 ]; then
    echo "⚠️  WARNING: Found $cuda_errors CUDA errors"
else
    echo "✅ No CUDA errors"
fi

echo ""

# ============================================================================
# 5. Recent Log Activity
# ============================================================================
echo "5. RECENT LOG ACTIVITY (Last 5 minutes)"
echo "--------------------------------------------------------------------------------"

recent_logs=$(find results/*/logs -name "*.log" -mmin -5 2>/dev/null)
if [ -n "$recent_logs" ]; then
    echo "Recently updated logs:"
    for log in $recent_logs; do
        echo "  - $(basename $log)"
        # Show last line of each recent log
        tail -n 1 "$log" 2>/dev/null | sed 's/^/    → /'
    done
else
    echo "No logs updated in last 5 minutes"
fi

echo ""

# ============================================================================
# 6. Estimated Progress
# ============================================================================
echo "6. OVERALL PROGRESS"
echo "--------------------------------------------------------------------------------"

total_submitted=33
total_success=$((dna_success + mnist_success + eeg_success))
total_in_queue=$((dna_pending + dna_running + mnist_pending + mnist_running + eeg_pending + eeg_running))

echo "Submitted: $total_submitted experiments"
echo "Completed: $total_success/$total_submitted ($(( (total_success * 100) / total_submitted ))%)"
echo "In queue:  $total_in_queue/$total_submitted"
echo ""

if [ $total_success -eq $total_submitted ]; then
    echo "🎉 ALL EXPERIMENTS COMPLETED!"
    echo ""
    echo "Next step: Run comprehensive analysis"
    echo "  cd /pscratch/sd/j/junghoon"
    echo "  python scripts/analyze_all_experiments.py > results/final_analysis_report.txt"
elif [ $total_in_queue -eq 0 ] && [ $total_success -lt $total_submitted ]; then
    echo "⚠️  Some experiments finished but not all successful"
    echo "Check error logs above for details"
elif [ $running_jobs -gt 0 ]; then
    echo "⏳ Experiments currently running..."
    echo "Check back in 1-2 hours for MNIST, 4-6 hours for DNA, 8-10 hours for EEG"
else
    echo "⏳ All experiments pending in queue, waiting for resources..."
fi

echo ""
echo "================================================================================"
echo "Monitoring complete. Re-run this script to check for updates."
echo "================================================================================"
echo ""
echo "Useful commands:"
echo "  Watch a specific log:  tail -f results/mnist_results/logs/mnist_classical_hydra_seed2024.log"
echo "  Check job details:     scontrol show job <JOB_ID>"
echo "  Cancel a job:          scancel <JOB_ID>"
echo ""

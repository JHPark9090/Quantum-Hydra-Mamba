#!/bin/bash
# Submit all Forrelation experiments for Quantum Advantage Testing
# This follows the test plan: Phase 3 (Silver) and Phase 4 (Gold)

echo "================================================================================"
echo "SUBMITTING FORRELATION EXPERIMENTS (QUANTUM ADVANTAGE TEST)"
echo "================================================================================"
echo ""

# Count total jobs
total_jobs=$(ls job_scripts/forrelation/*.sh 2>/dev/null | wc -l)

if [ $total_jobs -eq 0 ]; then
    echo "ERROR: No job scripts found in job_scripts/forrelation/"
    echo "Please run: python scripts/generate_forrelation_job_scripts.py first"
    exit 1
fi

echo "Total job scripts found: $total_jobs"
echo ""

# Ask for confirmation
read -p "Submit all $total_jobs jobs? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 0
fi

echo ""
echo "Submitting jobs..."
echo ""

# Counter
submitted=0
failed=0

# Submit all job scripts
for script in job_scripts/forrelation/*.sh; do
    job_name=$(basename "$script" .sh)

    # Submit job
    output=$(sbatch "$script" 2>&1)

    if [ $? -eq 0 ]; then
        job_id=$(echo "$output" | grep -oP 'Submitted batch job \K\d+')
        echo "[$((submitted+1))/$total_jobs] ✓ Submitted: $job_name (Job ID: $job_id)"
        ((submitted++))
    else
        echo "[$((submitted+failed+1))/$total_jobs] ✗ Failed: $job_name"
        echo "  Error: $output"
        ((failed++))
    fi
done

echo ""
echo "================================================================================"
echo "SUBMISSION COMPLETE"
echo "================================================================================"
echo "Successfully submitted: $submitted jobs"
echo "Failed submissions: $failed jobs"
echo ""

if [ $submitted -gt 0 ]; then
    echo "Monitor job status with:"
    echo "  squeue -u \$USER"
    echo "  watch -n 10 'squeue -u \$USER | grep forr_'"
    echo ""
    echo "View logs in:"
    echo "  results/forrelation_results/logs/"
    echo ""
    echo "Results will be saved to:"
    echo "  results/forrelation_results/*.json (metrics)"
    echo "  results/forrelation_results/*.pt (model checkpoints)"
fi

echo "================================================================================"

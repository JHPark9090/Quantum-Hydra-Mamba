#!/bin/bash
# Submit all individual synthetic benchmark jobs

cd jobs/synthetic

for script in syn_*.sh; do
    echo "Submitting $script"
    sbatch "$script"
    sleep 0.5  # Small delay to avoid overwhelming scheduler
done

echo "All jobs submitted!"

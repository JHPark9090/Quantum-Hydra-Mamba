#!/bin/bash
# Master submission script for all 30 EEG classification jobs
# Run all jobs in parallel (if you have GPU resources)
# Or submit in batches

echo "Submitting 30 EEG classification jobs..."
echo "================================================"

# Job 1: eeg_quantum_hydra_seed2024.sh
sbatch job_scripts/eeg/eeg_quantum_hydra_seed2024.sh
echo "Submitted: eeg_quantum_hydra_seed2024.sh"

# Job 2: eeg_quantum_hydra_seed2025.sh
sbatch job_scripts/eeg/eeg_quantum_hydra_seed2025.sh
echo "Submitted: eeg_quantum_hydra_seed2025.sh"

# Job 3: eeg_quantum_hydra_seed2026.sh
sbatch job_scripts/eeg/eeg_quantum_hydra_seed2026.sh
echo "Submitted: eeg_quantum_hydra_seed2026.sh"

# Job 4: eeg_quantum_hydra_seed2027.sh
sbatch job_scripts/eeg/eeg_quantum_hydra_seed2027.sh
echo "Submitted: eeg_quantum_hydra_seed2027.sh"

# Job 5: eeg_quantum_hydra_seed2028.sh
sbatch job_scripts/eeg/eeg_quantum_hydra_seed2028.sh
echo "Submitted: eeg_quantum_hydra_seed2028.sh"

# Job 6: eeg_quantum_hydra_hybrid_seed2024.sh
sbatch job_scripts/eeg/eeg_quantum_hydra_hybrid_seed2024.sh
echo "Submitted: eeg_quantum_hydra_hybrid_seed2024.sh"

# Pause to avoid overwhelming SLURM queue
echo "Pausing for 2 seconds..."
sleep 2

# Job 7: eeg_quantum_hydra_hybrid_seed2025.sh
sbatch job_scripts/eeg/eeg_quantum_hydra_hybrid_seed2025.sh
echo "Submitted: eeg_quantum_hydra_hybrid_seed2025.sh"

# Job 8: eeg_quantum_hydra_hybrid_seed2026.sh
sbatch job_scripts/eeg/eeg_quantum_hydra_hybrid_seed2026.sh
echo "Submitted: eeg_quantum_hydra_hybrid_seed2026.sh"

# Job 9: eeg_quantum_hydra_hybrid_seed2027.sh
sbatch job_scripts/eeg/eeg_quantum_hydra_hybrid_seed2027.sh
echo "Submitted: eeg_quantum_hydra_hybrid_seed2027.sh"

# Job 10: eeg_quantum_hydra_hybrid_seed2028.sh
sbatch job_scripts/eeg/eeg_quantum_hydra_hybrid_seed2028.sh
echo "Submitted: eeg_quantum_hydra_hybrid_seed2028.sh"

# Job 11: eeg_quantum_mamba_seed2024.sh
sbatch job_scripts/eeg/eeg_quantum_mamba_seed2024.sh
echo "Submitted: eeg_quantum_mamba_seed2024.sh"

# Job 12: eeg_quantum_mamba_seed2025.sh
sbatch job_scripts/eeg/eeg_quantum_mamba_seed2025.sh
echo "Submitted: eeg_quantum_mamba_seed2025.sh"

# Pause to avoid overwhelming SLURM queue
echo "Pausing for 2 seconds..."
sleep 2

# Job 13: eeg_quantum_mamba_seed2026.sh
sbatch job_scripts/eeg/eeg_quantum_mamba_seed2026.sh
echo "Submitted: eeg_quantum_mamba_seed2026.sh"

# Job 14: eeg_quantum_mamba_seed2027.sh
sbatch job_scripts/eeg/eeg_quantum_mamba_seed2027.sh
echo "Submitted: eeg_quantum_mamba_seed2027.sh"

# Job 15: eeg_quantum_mamba_seed2028.sh
sbatch job_scripts/eeg/eeg_quantum_mamba_seed2028.sh
echo "Submitted: eeg_quantum_mamba_seed2028.sh"

# Job 16: eeg_quantum_mamba_hybrid_seed2024.sh
sbatch job_scripts/eeg/eeg_quantum_mamba_hybrid_seed2024.sh
echo "Submitted: eeg_quantum_mamba_hybrid_seed2024.sh"

# Job 17: eeg_quantum_mamba_hybrid_seed2025.sh
sbatch job_scripts/eeg/eeg_quantum_mamba_hybrid_seed2025.sh
echo "Submitted: eeg_quantum_mamba_hybrid_seed2025.sh"

# Job 18: eeg_quantum_mamba_hybrid_seed2026.sh
sbatch job_scripts/eeg/eeg_quantum_mamba_hybrid_seed2026.sh
echo "Submitted: eeg_quantum_mamba_hybrid_seed2026.sh"

# Pause to avoid overwhelming SLURM queue
echo "Pausing for 2 seconds..."
sleep 2

# Job 19: eeg_quantum_mamba_hybrid_seed2027.sh
sbatch job_scripts/eeg/eeg_quantum_mamba_hybrid_seed2027.sh
echo "Submitted: eeg_quantum_mamba_hybrid_seed2027.sh"

# Job 20: eeg_quantum_mamba_hybrid_seed2028.sh
sbatch job_scripts/eeg/eeg_quantum_mamba_hybrid_seed2028.sh
echo "Submitted: eeg_quantum_mamba_hybrid_seed2028.sh"

# Job 21: eeg_classical_hydra_seed2024.sh
sbatch job_scripts/eeg/eeg_classical_hydra_seed2024.sh
echo "Submitted: eeg_classical_hydra_seed2024.sh"

# Job 22: eeg_classical_hydra_seed2025.sh
sbatch job_scripts/eeg/eeg_classical_hydra_seed2025.sh
echo "Submitted: eeg_classical_hydra_seed2025.sh"

# Job 23: eeg_classical_hydra_seed2026.sh
sbatch job_scripts/eeg/eeg_classical_hydra_seed2026.sh
echo "Submitted: eeg_classical_hydra_seed2026.sh"

# Job 24: eeg_classical_hydra_seed2027.sh
sbatch job_scripts/eeg/eeg_classical_hydra_seed2027.sh
echo "Submitted: eeg_classical_hydra_seed2027.sh"

# Pause to avoid overwhelming SLURM queue
echo "Pausing for 2 seconds..."
sleep 2

# Job 25: eeg_classical_hydra_seed2028.sh
sbatch job_scripts/eeg/eeg_classical_hydra_seed2028.sh
echo "Submitted: eeg_classical_hydra_seed2028.sh"

# Job 26: eeg_classical_mamba_seed2024.sh
sbatch job_scripts/eeg/eeg_classical_mamba_seed2024.sh
echo "Submitted: eeg_classical_mamba_seed2024.sh"

# Job 27: eeg_classical_mamba_seed2025.sh
sbatch job_scripts/eeg/eeg_classical_mamba_seed2025.sh
echo "Submitted: eeg_classical_mamba_seed2025.sh"

# Job 28: eeg_classical_mamba_seed2026.sh
sbatch job_scripts/eeg/eeg_classical_mamba_seed2026.sh
echo "Submitted: eeg_classical_mamba_seed2026.sh"

# Job 29: eeg_classical_mamba_seed2027.sh
sbatch job_scripts/eeg/eeg_classical_mamba_seed2027.sh
echo "Submitted: eeg_classical_mamba_seed2027.sh"

# Job 30: eeg_classical_mamba_seed2028.sh
sbatch job_scripts/eeg/eeg_classical_mamba_seed2028.sh
echo "Submitted: eeg_classical_mamba_seed2028.sh"


echo "================================================"
echo "All jobs submitted!"
echo "Monitor progress with: tail -f results/eeg_results/logs/*.log"
echo "Or check individual logs in: results/eeg_results/logs/"

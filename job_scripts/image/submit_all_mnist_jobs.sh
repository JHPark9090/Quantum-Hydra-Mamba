#!/bin/bash
# Master submission script for all 30 MNIST classification jobs
# Run all jobs in parallel (if you have GPU resources)
# Or submit in batches

echo "Submitting 30 MNIST classification jobs..."
echo "================================================"

# Job 1: mnist_quantum_hydra_seed2024.sh
sbatch job_scripts/mnist_quantum_hydra_seed2024.sh
echo "Submitted: mnist_quantum_hydra_seed2024.sh"

# Job 2: mnist_quantum_hydra_seed2025.sh
sbatch job_scripts/mnist_quantum_hydra_seed2025.sh
echo "Submitted: mnist_quantum_hydra_seed2025.sh"

# Job 3: mnist_quantum_hydra_seed2026.sh
sbatch job_scripts/mnist_quantum_hydra_seed2026.sh
echo "Submitted: mnist_quantum_hydra_seed2026.sh"

# Job 4: mnist_quantum_hydra_seed2027.sh
sbatch job_scripts/mnist_quantum_hydra_seed2027.sh
echo "Submitted: mnist_quantum_hydra_seed2027.sh"

# Job 5: mnist_quantum_hydra_seed2028.sh
sbatch job_scripts/mnist_quantum_hydra_seed2028.sh
echo "Submitted: mnist_quantum_hydra_seed2028.sh"

# Job 6: mnist_quantum_hydra_hybrid_seed2024.sh
sbatch job_scripts/mnist_quantum_hydra_hybrid_seed2024.sh
echo "Submitted: mnist_quantum_hydra_hybrid_seed2024.sh"

# Pause to avoid overwhelming SLURM queue
echo "Pausing for 2 seconds..."
sleep 2

# Job 7: mnist_quantum_hydra_hybrid_seed2025.sh
sbatch job_scripts/mnist_quantum_hydra_hybrid_seed2025.sh
echo "Submitted: mnist_quantum_hydra_hybrid_seed2025.sh"

# Job 8: mnist_quantum_hydra_hybrid_seed2026.sh
sbatch job_scripts/mnist_quantum_hydra_hybrid_seed2026.sh
echo "Submitted: mnist_quantum_hydra_hybrid_seed2026.sh"

# Job 9: mnist_quantum_hydra_hybrid_seed2027.sh
sbatch job_scripts/mnist_quantum_hydra_hybrid_seed2027.sh
echo "Submitted: mnist_quantum_hydra_hybrid_seed2027.sh"

# Job 10: mnist_quantum_hydra_hybrid_seed2028.sh
sbatch job_scripts/mnist_quantum_hydra_hybrid_seed2028.sh
echo "Submitted: mnist_quantum_hydra_hybrid_seed2028.sh"

# Job 11: mnist_quantum_mamba_seed2024.sh
sbatch job_scripts/mnist_quantum_mamba_seed2024.sh
echo "Submitted: mnist_quantum_mamba_seed2024.sh"

# Job 12: mnist_quantum_mamba_seed2025.sh
sbatch job_scripts/mnist_quantum_mamba_seed2025.sh
echo "Submitted: mnist_quantum_mamba_seed2025.sh"

# Pause to avoid overwhelming SLURM queue
echo "Pausing for 2 seconds..."
sleep 2

# Job 13: mnist_quantum_mamba_seed2026.sh
sbatch job_scripts/mnist_quantum_mamba_seed2026.sh
echo "Submitted: mnist_quantum_mamba_seed2026.sh"

# Job 14: mnist_quantum_mamba_seed2027.sh
sbatch job_scripts/mnist_quantum_mamba_seed2027.sh
echo "Submitted: mnist_quantum_mamba_seed2027.sh"

# Job 15: mnist_quantum_mamba_seed2028.sh
sbatch job_scripts/mnist_quantum_mamba_seed2028.sh
echo "Submitted: mnist_quantum_mamba_seed2028.sh"

# Job 16: mnist_quantum_mamba_hybrid_seed2024.sh
sbatch job_scripts/mnist_quantum_mamba_hybrid_seed2024.sh
echo "Submitted: mnist_quantum_mamba_hybrid_seed2024.sh"

# Job 17: mnist_quantum_mamba_hybrid_seed2025.sh
sbatch job_scripts/mnist_quantum_mamba_hybrid_seed2025.sh
echo "Submitted: mnist_quantum_mamba_hybrid_seed2025.sh"

# Job 18: mnist_quantum_mamba_hybrid_seed2026.sh
sbatch job_scripts/mnist_quantum_mamba_hybrid_seed2026.sh
echo "Submitted: mnist_quantum_mamba_hybrid_seed2026.sh"

# Pause to avoid overwhelming SLURM queue
echo "Pausing for 2 seconds..."
sleep 2

# Job 19: mnist_quantum_mamba_hybrid_seed2027.sh
sbatch job_scripts/mnist_quantum_mamba_hybrid_seed2027.sh
echo "Submitted: mnist_quantum_mamba_hybrid_seed2027.sh"

# Job 20: mnist_quantum_mamba_hybrid_seed2028.sh
sbatch job_scripts/mnist_quantum_mamba_hybrid_seed2028.sh
echo "Submitted: mnist_quantum_mamba_hybrid_seed2028.sh"

# Job 21: mnist_classical_hydra_seed2024.sh
sbatch job_scripts/mnist_classical_hydra_seed2024.sh
echo "Submitted: mnist_classical_hydra_seed2024.sh"

# Job 22: mnist_classical_hydra_seed2025.sh
sbatch job_scripts/mnist_classical_hydra_seed2025.sh
echo "Submitted: mnist_classical_hydra_seed2025.sh"

# Job 23: mnist_classical_hydra_seed2026.sh
sbatch job_scripts/mnist_classical_hydra_seed2026.sh
echo "Submitted: mnist_classical_hydra_seed2026.sh"

# Job 24: mnist_classical_hydra_seed2027.sh
sbatch job_scripts/mnist_classical_hydra_seed2027.sh
echo "Submitted: mnist_classical_hydra_seed2027.sh"

# Pause to avoid overwhelming SLURM queue
echo "Pausing for 2 seconds..."
sleep 2

# Job 25: mnist_classical_hydra_seed2028.sh
sbatch job_scripts/mnist_classical_hydra_seed2028.sh
echo "Submitted: mnist_classical_hydra_seed2028.sh"

# Job 26: mnist_classical_mamba_seed2024.sh
sbatch job_scripts/mnist_classical_mamba_seed2024.sh
echo "Submitted: mnist_classical_mamba_seed2024.sh"

# Job 27: mnist_classical_mamba_seed2025.sh
sbatch job_scripts/mnist_classical_mamba_seed2025.sh
echo "Submitted: mnist_classical_mamba_seed2025.sh"

# Job 28: mnist_classical_mamba_seed2026.sh
sbatch job_scripts/mnist_classical_mamba_seed2026.sh
echo "Submitted: mnist_classical_mamba_seed2026.sh"

# Job 29: mnist_classical_mamba_seed2027.sh
sbatch job_scripts/mnist_classical_mamba_seed2027.sh
echo "Submitted: mnist_classical_mamba_seed2027.sh"

# Job 30: mnist_classical_mamba_seed2028.sh
sbatch job_scripts/mnist_classical_mamba_seed2028.sh
echo "Submitted: mnist_classical_mamba_seed2028.sh"


echo "================================================"
echo "All jobs submitted!"
echo "Monitor progress with: tail -f results/mnist_results/logs/*.log"
echo "Or check individual logs in: results/mnist_results/logs/"

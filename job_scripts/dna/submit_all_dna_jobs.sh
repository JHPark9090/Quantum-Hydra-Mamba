#!/bin/bash
# Master submission script for all 30 DNA sequence classification jobs
# Run all jobs in parallel (if you have GPU resources)
# Or submit in batches

echo "Submitting 30 DNA sequence classification jobs..."
echo "================================================"

# Job 1: dna_quantum_hydra_seed2024.sh
sbatch job_scripts/dna/dna_quantum_hydra_seed2024.sh
echo "Submitted: dna_quantum_hydra_seed2024.sh"

# Job 2: dna_quantum_hydra_seed2025.sh
sbatch job_scripts/dna/dna_quantum_hydra_seed2025.sh
echo "Submitted: dna_quantum_hydra_seed2025.sh"

# Job 3: dna_quantum_hydra_seed2026.sh
sbatch job_scripts/dna/dna_quantum_hydra_seed2026.sh
echo "Submitted: dna_quantum_hydra_seed2026.sh"

# Job 4: dna_quantum_hydra_seed2027.sh
sbatch job_scripts/dna/dna_quantum_hydra_seed2027.sh
echo "Submitted: dna_quantum_hydra_seed2027.sh"

# Job 5: dna_quantum_hydra_seed2028.sh
sbatch job_scripts/dna/dna_quantum_hydra_seed2028.sh
echo "Submitted: dna_quantum_hydra_seed2028.sh"

# Job 6: dna_quantum_hydra_hybrid_seed2024.sh
sbatch job_scripts/dna/dna_quantum_hydra_hybrid_seed2024.sh
echo "Submitted: dna_quantum_hydra_hybrid_seed2024.sh"

# Pause to avoid overwhelming SLURM queue
echo "Pausing for 2 seconds..."
sleep 2

# Job 7: dna_quantum_hydra_hybrid_seed2025.sh
sbatch job_scripts/dna/dna_quantum_hydra_hybrid_seed2025.sh
echo "Submitted: dna_quantum_hydra_hybrid_seed2025.sh"

# Job 8: dna_quantum_hydra_hybrid_seed2026.sh
sbatch job_scripts/dna/dna_quantum_hydra_hybrid_seed2026.sh
echo "Submitted: dna_quantum_hydra_hybrid_seed2026.sh"

# Job 9: dna_quantum_hydra_hybrid_seed2027.sh
sbatch job_scripts/dna/dna_quantum_hydra_hybrid_seed2027.sh
echo "Submitted: dna_quantum_hydra_hybrid_seed2027.sh"

# Job 10: dna_quantum_hydra_hybrid_seed2028.sh
sbatch job_scripts/dna/dna_quantum_hydra_hybrid_seed2028.sh
echo "Submitted: dna_quantum_hydra_hybrid_seed2028.sh"

# Job 11: dna_quantum_mamba_seed2024.sh
sbatch job_scripts/dna/dna_quantum_mamba_seed2024.sh
echo "Submitted: dna_quantum_mamba_seed2024.sh"

# Job 12: dna_quantum_mamba_seed2025.sh
sbatch job_scripts/dna/dna_quantum_mamba_seed2025.sh
echo "Submitted: dna_quantum_mamba_seed2025.sh"

# Pause to avoid overwhelming SLURM queue
echo "Pausing for 2 seconds..."
sleep 2

# Job 13: dna_quantum_mamba_seed2026.sh
sbatch job_scripts/dna/dna_quantum_mamba_seed2026.sh
echo "Submitted: dna_quantum_mamba_seed2026.sh"

# Job 14: dna_quantum_mamba_seed2027.sh
sbatch job_scripts/dna/dna_quantum_mamba_seed2027.sh
echo "Submitted: dna_quantum_mamba_seed2027.sh"

# Job 15: dna_quantum_mamba_seed2028.sh
sbatch job_scripts/dna/dna_quantum_mamba_seed2028.sh
echo "Submitted: dna_quantum_mamba_seed2028.sh"

# Job 16: dna_quantum_mamba_hybrid_seed2024.sh
sbatch job_scripts/dna/dna_quantum_mamba_hybrid_seed2024.sh
echo "Submitted: dna_quantum_mamba_hybrid_seed2024.sh"

# Job 17: dna_quantum_mamba_hybrid_seed2025.sh
sbatch job_scripts/dna/dna_quantum_mamba_hybrid_seed2025.sh
echo "Submitted: dna_quantum_mamba_hybrid_seed2025.sh"

# Job 18: dna_quantum_mamba_hybrid_seed2026.sh
sbatch job_scripts/dna/dna_quantum_mamba_hybrid_seed2026.sh
echo "Submitted: dna_quantum_mamba_hybrid_seed2026.sh"

# Pause to avoid overwhelming SLURM queue
echo "Pausing for 2 seconds..."
sleep 2

# Job 19: dna_quantum_mamba_hybrid_seed2027.sh
sbatch job_scripts/dna/dna_quantum_mamba_hybrid_seed2027.sh
echo "Submitted: dna_quantum_mamba_hybrid_seed2027.sh"

# Job 20: dna_quantum_mamba_hybrid_seed2028.sh
sbatch job_scripts/dna/dna_quantum_mamba_hybrid_seed2028.sh
echo "Submitted: dna_quantum_mamba_hybrid_seed2028.sh"

# Job 21: dna_classical_hydra_seed2024.sh
sbatch job_scripts/dna/dna_classical_hydra_seed2024.sh
echo "Submitted: dna_classical_hydra_seed2024.sh"

# Job 22: dna_classical_hydra_seed2025.sh
sbatch job_scripts/dna/dna_classical_hydra_seed2025.sh
echo "Submitted: dna_classical_hydra_seed2025.sh"

# Job 23: dna_classical_hydra_seed2026.sh
sbatch job_scripts/dna/dna_classical_hydra_seed2026.sh
echo "Submitted: dna_classical_hydra_seed2026.sh"

# Job 24: dna_classical_hydra_seed2027.sh
sbatch job_scripts/dna/dna_classical_hydra_seed2027.sh
echo "Submitted: dna_classical_hydra_seed2027.sh"

# Pause to avoid overwhelming SLURM queue
echo "Pausing for 2 seconds..."
sleep 2

# Job 25: dna_classical_hydra_seed2028.sh
sbatch job_scripts/dna/dna_classical_hydra_seed2028.sh
echo "Submitted: dna_classical_hydra_seed2028.sh"

# Job 26: dna_classical_mamba_seed2024.sh
sbatch job_scripts/dna/dna_classical_mamba_seed2024.sh
echo "Submitted: dna_classical_mamba_seed2024.sh"

# Job 27: dna_classical_mamba_seed2025.sh
sbatch job_scripts/dna/dna_classical_mamba_seed2025.sh
echo "Submitted: dna_classical_mamba_seed2025.sh"

# Job 28: dna_classical_mamba_seed2026.sh
sbatch job_scripts/dna/dna_classical_mamba_seed2026.sh
echo "Submitted: dna_classical_mamba_seed2026.sh"

# Job 29: dna_classical_mamba_seed2027.sh
sbatch job_scripts/dna/dna_classical_mamba_seed2027.sh
echo "Submitted: dna_classical_mamba_seed2027.sh"

# Job 30: dna_classical_mamba_seed2028.sh
sbatch job_scripts/dna/dna_classical_mamba_seed2028.sh
echo "Submitted: dna_classical_mamba_seed2028.sh"


echo "================================================"
echo "All jobs submitted!"
echo "Monitor progress with: tail -f results/dna_results/logs/*.log"
echo "Or check individual logs in: results/dna_results/logs/"

#!/bin/bash
# Submit all 9 jobs for model 2c (ClassicalHydraQuantumSSM)
echo "Submitting 9 jobs for 2c..."

sbatch ./jobs/ablation_eeg/40Hz/abl_2c_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2c_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2c_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2c_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2c_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2c_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2c_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2c_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2c_160Hz_s2026.sh

echo "Submitted 9 jobs for model 2c"

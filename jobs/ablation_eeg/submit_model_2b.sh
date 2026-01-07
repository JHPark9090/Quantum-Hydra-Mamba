#!/bin/bash
# Submit all 9 jobs for model 2b (ClassicalMambaQuantumSSM)
echo "Submitting 9 jobs for 2b..."

sbatch ./jobs/ablation_eeg/40Hz/abl_2b_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2b_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2b_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2b_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2b_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2b_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2b_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2b_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2b_160Hz_s2026.sh

echo "Submitted 9 jobs for model 2b"

#!/bin/bash
# Submit all 27 jobs for group2
# Models: 2a, 2b, 2c
echo "Submitting 27 jobs for group2..."

sbatch ./jobs/ablation_eeg/40Hz/abl_2a_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2a_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2a_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2a_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2a_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2a_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2a_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2a_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2a_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2b_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2b_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2b_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2b_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2b_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2b_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2b_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2b_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2b_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2c_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2c_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_2c_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2c_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2c_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_2c_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2c_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2c_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_2c_160Hz_s2026.sh

echo "Submitted 27 jobs for group2"

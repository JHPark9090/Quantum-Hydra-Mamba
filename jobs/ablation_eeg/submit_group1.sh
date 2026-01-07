#!/bin/bash
# Submit all 27 jobs for group1
# Models: 1a, 1b, 1c
echo "Submitting 27 jobs for group1..."

sbatch ./jobs/ablation_eeg/40Hz/abl_1a_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1a_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1a_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1a_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1a_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1a_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1a_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1a_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1a_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1b_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1b_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1b_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1b_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1b_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1b_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1b_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1b_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1b_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1c_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1c_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_1c_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1c_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1c_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_1c_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1c_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1c_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_1c_160Hz_s2026.sh

echo "Submitted 27 jobs for group1"

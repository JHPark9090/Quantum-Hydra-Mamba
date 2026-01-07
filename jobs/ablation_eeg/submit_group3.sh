#!/bin/bash
# Submit all 27 jobs for group3
# Models: 3a, 3b, 3c
echo "Submitting 27 jobs for group3..."

sbatch ./jobs/ablation_eeg/40Hz/abl_3a_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3a_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3a_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3a_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3a_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3a_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3a_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3a_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3a_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3b_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3b_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3b_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3b_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3b_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3b_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3b_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3b_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3b_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3c_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3c_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3c_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3c_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3c_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3c_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3c_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3c_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3c_160Hz_s2026.sh

echo "Submitted 27 jobs for group3"

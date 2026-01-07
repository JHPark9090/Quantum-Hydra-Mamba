#!/bin/bash
# Submit all 27 jobs for group4
# Models: 4a, 4b, 4c
echo "Submitting 27 jobs for group4..."

sbatch ./jobs/ablation_eeg/40Hz/abl_4a_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4a_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4a_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4a_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4a_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4a_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4a_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4a_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4a_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4b_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4b_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4b_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4b_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4b_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4b_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4b_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4b_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4b_160Hz_s2026.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4c_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4c_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4c_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4c_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4c_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4c_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4c_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4c_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4c_160Hz_s2026.sh

echo "Submitted 27 jobs for group4"

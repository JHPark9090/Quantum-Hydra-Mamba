#!/bin/bash
# Submit all 9 jobs for model 3a (ClassicalTransformer)
echo "Submitting 9 jobs for 3a..."

sbatch ./jobs/ablation_eeg/40Hz/abl_3a_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3a_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_3a_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3a_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3a_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_3a_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3a_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3a_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_3a_160Hz_s2026.sh

echo "Submitted 9 jobs for model 3a"

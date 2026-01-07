#!/bin/bash
# Submit all 9 jobs for model 4b (QuantumMambaE2E)
echo "Submitting 9 jobs for 4b..."

sbatch ./jobs/ablation_eeg/40Hz/abl_4b_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4b_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4b_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4b_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4b_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4b_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4b_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4b_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4b_160Hz_s2026.sh

echo "Submitted 9 jobs for model 4b"

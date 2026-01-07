#!/bin/bash
# Submit all 9 jobs for model 4c (QuantumHydraE2E)
echo "Submitting 9 jobs for 4c..."

sbatch ./jobs/ablation_eeg/40Hz/abl_4c_40Hz_s2024.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4c_40Hz_s2025.sh
sbatch ./jobs/ablation_eeg/40Hz/abl_4c_40Hz_s2026.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4c_80Hz_s2024.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4c_80Hz_s2025.sh
sbatch ./jobs/ablation_eeg/80Hz/abl_4c_80Hz_s2026.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4c_160Hz_s2024.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4c_160Hz_s2025.sh
sbatch ./jobs/ablation_eeg/160Hz/abl_4c_160Hz_s2026.sh

echo "Submitted 9 jobs for model 4c"

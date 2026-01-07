#!/bin/bash
# Generate all Forrelation datasets for Quantum Advantage Testing
# Following the test plan in Quantum_Advantage_Test_Plan.md

echo "================================================================================"
echo "GENERATING FORRELATION DATASETS FOR QUANTUM ADVANTAGE TESTING"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p forrelation_data

# Phase 1: Tuning Dataset
echo "Phase 1: Generating Tuning Dataset..."
python data_loaders/generate_forrelation_dataset.py \
    --num_pairs 2000 \
    --n_bits 6 \
    --seq_len 80 \
    --filename forrelation_data/tuning_dataset.pt

echo ""
echo "Phase 1 complete!"
echo ""

# Phase 2: Challenge Dataset (Bronze Standard)
echo "Phase 2: Generating Challenge Dataset (Bronze Standard)..."
python data_loaders/generate_forrelation_dataset.py \
    --num_pairs 5000 \
    --n_bits 7 \
    --seq_len 120 \
    --filename forrelation_data/challenge_dataset.pt

echo ""
echo "Phase 2 complete!"
echo ""

# Phase 3: Sample Efficiency Test (Silver Standard)
# Test varying sequence lengths with fixed complexity
echo "Phase 3: Generating Sample Efficiency Datasets (Silver Standard)..."
echo "  n_bits=6 (fixed), varying seq_len"

for L in 20 40 80 160; do
    echo "  Generating L=$L..."
    python data_loaders/generate_forrelation_dataset.py \
        --num_pairs 3000 \
        --n_bits 6 \
        --seq_len $L \
        --filename forrelation_data/forrelation_L${L}.pt
done

echo ""
echo "Phase 3 complete!"
echo ""

# Phase 4: Scaling Test (Gold Standard)
# Test varying sequence lengths with higher complexity
echo "Phase 4: Generating Scaling Datasets (Gold Standard)..."
echo "  n_bits=8 (harder), varying seq_len"

for L in 40 80 160 320; do
    echo "  Generating n8_L=$L..."
    python data_loaders/generate_forrelation_dataset.py \
        --num_pairs 3000 \
        --n_bits 8 \
        --seq_len $L \
        --filename forrelation_data/forrelation_n8_L${L}.pt
done

echo ""
echo "Phase 4 complete!"
echo ""

echo "================================================================================"
echo "ALL DATASETS GENERATED SUCCESSFULLY!"
echo "================================================================================"
echo "Summary of generated datasets:"
echo ""
ls -lh forrelation_data/*.pt
echo ""
echo "Total datasets: $(ls forrelation_data/*.pt | wc -l)"
echo ""
echo "Next steps:"
echo "  1. Generate job scripts:"
echo "     python scripts/generate_forrelation_job_scripts.py"
echo ""
echo "  2. Submit jobs for training:"
echo "     bash job_scripts/submit_forrelation_experiments.sh"
echo "================================================================================"

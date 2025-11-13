#!/bin/bash
# Generate Forrelation datasets for Quantum Advantage Testing

echo "================================================================================"
echo "GENERATING FORRELATION DATASETS"
echo "================================================================================"
echo ""

mkdir -p forrelation_data

echo "Phase 3: Sample Efficiency Test (Silver Standard)"
echo "  Generating datasets with n_bits=6, varying sequence lengths..."

for L in 20 40 80 160; do
    echo "  Creating dataset with L=$L..."
    python datasets/generate_forrelation_dataset.py \
        --num_pairs 3000 \
        --n_bits 6 \
        --seq_len $L \
        --filename forrelation_L${L}.pt

    mv forrelation_L${L}.pt forrelation_data/
done

echo ""
echo "Phase 4: Scaling Test (Gold Standard)"
echo "  Generating datasets with n_bits=8, varying sequence lengths..."

for L in 40 80 160 320; do
    echo "  Creating dataset with L=$L..."
    python datasets/generate_forrelation_dataset.py \
        --num_pairs 3000 \
        --n_bits 8 \
        --seq_len $L \
        --filename forrelation_n8_L${L}.pt

    mv forrelation_n8_L${L}.pt forrelation_data/
done

echo ""
echo "================================================================================"
echo "DATASET GENERATION COMPLETE!"
echo "================================================================================"
echo "Generated datasets:"
ls -lh forrelation_data/
echo ""
echo "Next step: Run experiments with scripts/run_all_forrelation_experiments.sh"
echo "================================================================================"

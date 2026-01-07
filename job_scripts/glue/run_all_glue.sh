#!/bin/bash
# ================================================================================
# Master Script: Submit All 9 GLUE Benchmark Experiments
#
# This script submits SLURM jobs for ALL 9 GLUE tasks.
# Each job trains 3 models: QuantumHydraGLUE, QuantumMambaGLUE, LSTMBaseline
#
# Total: 9 tasks × 3 models = 27 model training runs
# ================================================================================

echo "=============================================="
echo "Submitting ALL GLUE Benchmark Experiments"
echo "Date: $(date)"
echo "=============================================="

# Create directories
mkdir -p logs
mkdir -p glue_results

# ================================================================================
# Tier 1: Small/Fast Tasks (< 6 hours each)
# ================================================================================
echo ""
echo "[Tier 1] Small/Fast Tasks"
echo "========================="

# CoLA - Linguistic Acceptability (8.5k samples)
echo "1. Submitting CoLA (8.5k samples, ~4-6 hours)..."
JOB_COLA=$(sbatch --parsable run_glue_cola.sh)
echo "   Job ID: $JOB_COLA"

# MRPC - Paraphrase Detection (3.7k samples)
echo "2. Submitting MRPC (3.7k samples, ~3-5 hours)..."
JOB_MRPC=$(sbatch --parsable run_glue_mrpc.sh)
echo "   Job ID: $JOB_MRPC"

# RTE - Textual Entailment (2.5k samples)
echo "3. Submitting RTE (2.5k samples, ~2-4 hours)..."
JOB_RTE=$(sbatch --parsable run_glue_rte.sh)
echo "   Job ID: $JOB_RTE"

# WNLI - Winograd NLI (634 samples - very small but difficult)
echo "4. Submitting WNLI (634 samples, ~1-2 hours)..."
JOB_WNLI=$(sbatch --parsable run_glue_wnli.sh)
echo "   Job ID: $JOB_WNLI"

# STS-B - Semantic Similarity Regression (5.7k samples)
echo "5. Submitting STS-B (5.7k samples, ~4-6 hours)..."
JOB_STSB=$(sbatch --parsable run_glue_stsb.sh)
echo "   Job ID: $JOB_STSB"

# ================================================================================
# Tier 2: Medium Tasks (6-12 hours each)
# ================================================================================
echo ""
echo "[Tier 2] Medium Tasks"
echo "====================="

# SST-2 - Sentiment Analysis (67k samples)
echo "6. Submitting SST-2 (67k samples, ~8-12 hours)..."
JOB_SST2=$(sbatch --parsable run_glue_sst2.sh)
echo "   Job ID: $JOB_SST2"

# ================================================================================
# Tier 3: Large Tasks (12-24 hours each)
# ================================================================================
echo ""
echo "[Tier 3] Large Tasks"
echo "===================="

# QNLI - Question NLI (105k samples)
echo "7. Submitting QNLI (105k samples, ~12-18 hours)..."
JOB_QNLI=$(sbatch --parsable run_glue_qnli.sh)
echo "   Job ID: $JOB_QNLI"

# QQP - Quora Question Pairs (364k samples)
echo "8. Submitting QQP (364k samples, ~18-24 hours)..."
JOB_QQP=$(sbatch --parsable run_glue_qqp.sh)
echo "   Job ID: $JOB_QQP"

# MNLI - Multi-Genre NLI (393k samples - LARGEST)
echo "9. Submitting MNLI (393k samples, ~20-24 hours)..."
JOB_MNLI=$(sbatch --parsable run_glue_mnli.sh)
echo "   Job ID: $JOB_MNLI"

# ================================================================================
# Summary
# ================================================================================
echo ""
echo "=============================================="
echo "ALL 9 GLUE TASKS SUBMITTED"
echo "=============================================="
echo ""
echo "Tier 1 (Small/Fast):"
echo "  1. CoLA:  $JOB_COLA"
echo "  2. MRPC:  $JOB_MRPC"
echo "  3. RTE:   $JOB_RTE"
echo "  4. WNLI:  $JOB_WNLI"
echo "  5. STS-B: $JOB_STSB"
echo ""
echo "Tier 2 (Medium):"
echo "  6. SST-2: $JOB_SST2"
echo ""
echo "Tier 3 (Large):"
echo "  7. QNLI:  $JOB_QNLI"
echo "  8. QQP:   $JOB_QQP"
echo "  9. MNLI:  $JOB_MNLI"
echo ""
echo "=============================================="
echo "Total Jobs: 9"
echo "Models per job: 3 (QuantumHydra, QuantumMamba, LSTM)"
echo "Total model runs: 27"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Check logs:   tail -f logs/glue_*"
echo "=============================================="

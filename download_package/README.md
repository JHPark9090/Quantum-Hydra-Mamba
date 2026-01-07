# Quantum Hydra Mamba - Experiment Package

Complete experiment package for running quantum vs classical SSM/attention comparisons.

## Quick Start

```bash
# 1. Create conda environment
conda create -n qml_env python=3.11 -y
conda activate qml_env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run ablation study (Gated vs SSM vs Attention)
python scripts/run_full_comparison.py \
    --datasets demo_human_or_worm demo_coding_vs_intergenomic_seqs \
    --seeds 2024 2025 2026 \
    --n-epochs 30 \
    --device cuda
```

## Directory Structure

```
quantum_hydra_mamba/
├── models/                      # All model implementations
│   ├── QuantumSSM.py           # Quantum SSM (Mamba/Hydra)
│   ├── QuantumGatedRecurrence.py # Quantum Gated Recurrence
│   ├── QuantumAttention.py     # Quantum Attention
│   ├── TrueClassicalMamba.py   # Classical Mamba baseline
│   └── TrueClassicalHydra.py   # Classical Hydra baseline
├── data_loaders/                # Dataset loaders
│   ├── Load_Image_Datasets.py  # MNIST, CIFAR
│   ├── Load_PhysioNet_EEG_NoPrompt.py # EEG data
│   ├── Load_Genomic_Benchmarks.py # Genomic Benchmarks
│   ├── Load_GLUE.py            # GLUE NLP benchmark
│   └── forrelation_dataloader.py # Forrelation (quantum advantage)
├── scripts/                     # Run & aggregation scripts
│   ├── run_full_comparison.py  # Main comparison script
│   ├── run_ssm_genomic_comparison.py # Genomic Benchmarks runner
│   ├── run_single_model_*.py   # Single model runners
│   └── aggregate_*_results.py  # Result aggregation
├── job_scripts/                 # SLURM batch scripts
│   ├── eeg/                    # EEG experiment jobs
│   ├── image/                  # Image experiment jobs
│   ├── genomic/                # Genomic Benchmarks jobs
│   ├── glue/                   # GLUE benchmark jobs
│   └── forrelation/            # Forrelation experiment jobs
├── docs/                        # Documentation
│   ├── EXPERIMENTAL_PLAN_README.md
│   ├── QUANTUM_SSM_README.md
│   └── ...
└── StudyPlan.md                 # Research plan
```

## Experiments Overview

### 1. Ablation Study: Selective Mechanism Comparison

Compare three selective mechanisms:
- **Gated**: LSTM-style gates for selective forgetting
- **SSM**: State space models with selective scan
- **Attention**: Self-attention for global information mixing

**Run:**
```bash
python scripts/run_full_comparison.py \
    --datasets demo_human_or_worm demo_coding_vs_intergenomic_seqs drosophila_enhancers_stark \
    --seeds 2024 2025 2026 2027 2028 \
    --n-epochs 30 \
    --n-qubits 4 \
    --device cuda
```

### 2. Image Classification (MNIST)

```bash
# Single model
python scripts/run_single_model_mnist.py \
    --model-name quantum_hydra \
    --seed 2024 \
    --device cuda

# All models (via SLURM)
cd job_scripts/image
sbatch submit_all_mnist_jobs.sh
```

### 3. PhysioNet EEG Motor Imagery

```bash
# Single model
python scripts/run_single_model_eeg.py \
    --model-name quantum_mamba \
    --seed 2024 \
    --device cuda

# All models (via SLURM)
cd job_scripts/eeg
sbatch submit_all_eeg_jobs.sh
```

### 4. Genomic Benchmarks

Uses the `genomic-benchmarks` package with datasets:
- demo_human_or_worm (200 seq_len)
- demo_coding_vs_intergenomic_seqs (200 seq_len)
- drosophila_enhancers_stark (smaller)

```bash
# Using run_full_comparison.py (recommended)
python scripts/run_full_comparison.py \
    --datasets demo_human_or_worm \
    --seeds 2024 \
    --n-epochs 30 \
    --device cuda

# Using run_ssm_genomic_comparison.py
python scripts/run_ssm_genomic_comparison.py \
    --dataset demo_human_or_worm \
    --max-samples 500 \
    --n-epochs 30 \
    --seed 2024 \
    --device cuda

# Via SLURM
cd job_scripts/genomic
sbatch ssm_genomic_demo_human_or_worm.sh
```

### 5. GLUE Benchmark (NLP)

```bash
# Single task
python scripts/run_glue.py \
    --task cola \
    --model quantum_hydra \
    --seed 2024 \
    --device cuda

# All tasks (via SLURM)
cd job_scripts/glue
sbatch run_all_glue.sh
```

### 6. Forrelation (Quantum Advantage Test)

```bash
# Generate datasets
python data_loaders/generate_forrelation_dataset.py

# Run experiments
python scripts/run_single_model_forrelation.py \
    --model-name quantum_mamba \
    --seed 2024 \
    --device cuda
```

## Models

| Model | Type | Description |
|-------|------|-------------|
| `quantum_mamba_ssm` | Quantum | Unidirectional SSM with quantum encoding |
| `quantum_hydra_ssm` | Quantum | Bidirectional SSM with quantum encoding |
| `quantum_mamba_gated` | Quantum | Unidirectional gated recurrence with quantum |
| `quantum_hydra_gated` | Quantum | Bidirectional gated recurrence with quantum |
| `quantum_mamba_attention` | Quantum | Unidirectional attention with quantum |
| `quantum_hydra_attention` | Quantum | Bidirectional attention with quantum |
| `classical_mamba` | Classical | True Mamba implementation (Gu & Dao, 2024) |
| `classical_hydra` | Classical | True Hydra implementation (Hwang et al., 2024) |

## Configuration Options

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-qubits` | 4 | Number of qubits |
| `--qlcu-layers` | 2 | Quantum layers per block |
| `--d-model` | 64 | Model dimension |
| `--d-state` | 16 | State dimension (SSM) |
| `--n-layers` | 1 | Number of stacked blocks |
| `--n-epochs` | 30 | Training epochs |
| `--batch-size` | 16 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--seed` | 2024 | Random seed |
| `--device` | cuda | Device (cuda/cpu) |

## Statistical Analysis

### Running Multiple Seeds

All experiments should be run with 5 seeds: 2024, 2025, 2026, 2027, 2028

```bash
for seed in 2024 2025 2026 2027 2028; do
    python scripts/run_full_comparison.py \
        --datasets demo_human_or_worm \
        --seeds $seed \
        --n-epochs 30
done
```

### Aggregating Results

```bash
# Aggregate by experiment type
python scripts/aggregate_mnist_results.py
python scripts/aggregate_eeg_results.py
python scripts/aggregate_forrelation_results.py
```

### Statistical Tests

The aggregation scripts compute:
- Mean and standard deviation across seeds
- Paired t-test for model comparisons
- Wilcoxon signed-rank test (non-parametric)
- 95% confidence intervals

## Quantum Backend Configuration

### Default: lightning.qubit (CPU)

Fast CPU-based simulation, good for most use cases.

### Alternative: lightning.kokkos (OpenMP)

Multi-threaded CPU simulation. Used in `QuantumGatedRecurrence.py` to avoid hang issues with recurrence loops.

```python
# In models/QuantumGatedRecurrence.py
device = qml.device("lightning.kokkos", wires=n_qubits, batch_obs=True)
```

### Alternative: lightning.gpu (NVIDIA GPU)

GPU-accelerated simulation. Note: May hang with repeated QNode calls in loops.

```python
device = qml.device("lightning.gpu", wires=n_qubits, batch_obs=True)
```

## Adapting SLURM Scripts

The job scripts are written for NERSC Perlmutter. To adapt for other systems:

1. Change account/partition:
```bash
#SBATCH --account=your_account
#SBATCH --partition=gpu
```

2. Change module loads:
```bash
module load cuda/12.0
module load python/3.11
```

3. Change conda environment path:
```bash
source activate /path/to/your/conda/env
```

## Expected Results

### Ablation Study (Genomic Benchmarks)
- Expect quantum models to achieve 60-75% accuracy
- SSM typically outperforms Gated on sequence data
- Attention may struggle with very long sequences

### EEG Classification
- Expect 75-85% accuracy on motor imagery
- Quantum models competitive with classical

### Forrelation (Quantum Advantage)
- Silver Standard: 30-50% fewer samples to reach 90% accuracy
- Gold Standard: Advantage increases with problem complexity

## Troubleshooting

### CUDA Out of Memory
Reduce batch size or use gradient accumulation:
```bash
--batch-size 8
```

### Lightning.gpu Hangs
Switch to lightning.kokkos (already done in QuantumGatedRecurrence.py):
```python
device = qml.device("lightning.kokkos", wires=n_qubits)
```

### Device Mismatch (CPU/CUDA)
Ensure quantum outputs are moved to target device:
```python
quantum_output = quantum_output.to(self.device)
```

## References

1. Gu, A., & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
2. Hwang, W., Kim, M., Zhang, X., & Song, H. (2024). Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers.
3. Aaronson, S., & Ambainis, A. (2015). Forrelation: A Problem that Optimally Separates Quantum from Classical Computing.

## License

Research use only.

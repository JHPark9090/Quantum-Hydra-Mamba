# Quantum Hydra/Mamba - Ablation Study Framework

Comprehensive 2x2x3 factorial ablation study comparing quantum-classical hybrid architectures for sequence classification.

**Last Updated:** January 7, 2026

## Overview

This repository implements a rigorous ablation study to isolate quantum contributions in neural sequence models:

- **Feature Extraction**: Quantum vs Classical
- **Mixing Mechanism**: Quantum vs Classical
- **Architecture Type**: Transformer vs Mamba vs Hydra

**Total Models:** 16 (expanded from original 12 to include true quantum superposition variants)

## Ablation Study Design

### Model Groups

| Group | Features | Mixing | Models | Description |
|-------|----------|--------|--------|-------------|
| **1** | Quantum | Classical | 1a, 1b, 1c | Quantum feature extraction → Classical mixing |
| **2** | Classical | Quantum | 2a-2e | Classical features → Quantum mixing |
| **3** | Classical | Classical | 3a, 3b, 3c | Pure classical baselines |
| **4** | Quantum | Quantum | 4a-4e | End-to-end quantum (E2E) |

### Complete Model Registry (16 Models)

| ID | Model Name | Group | Features | Mixing | Architecture |
|----|------------|-------|----------|--------|--------------|
| **1a** | QuantumTransformer | 1 | Quantum | Classical | Transformer |
| **1b** | QuantumMambaSSM | 1 | Quantum | Classical | Mamba |
| **1c** | QuantumHydraSSM | 1 | Quantum | Classical | Hydra |
| **2a** | ClassicalQuantumAttention | 2 | Classical | Quantum | Transformer |
| **2b** | ClassicalMambaQuantumSSM | 2 | Classical | Quantum | Mamba |
| **2c** | ClassicalHydraQuantumSSM | 2 | Classical | Quantum | Hydra |
| **2d** | QuantumMambaHydraSSM | 2 | Classical | Quantum Superposition | Mamba |
| **2e** | QuantumHydraHydraSSM | 2 | Classical | Quantum Superposition | Hydra |
| **3a** | ClassicalTransformer | 3 | Classical | Classical | Transformer |
| **3b** | TrueClassicalMamba | 3 | Classical | Classical | Mamba |
| **3c** | TrueClassicalHydra | 3 | Classical | Classical | Hydra |
| **4a** | QuantumTransformerE2E | 4 | Quantum | Quantum | Transformer |
| **4b** | QuantumMambaE2E | 4 | Quantum | Quantum | Mamba |
| **4c** | QuantumHydraE2E | 4 | Quantum | Quantum | Hydra |
| **4d** | QuantumMambaE2ESuperposition | 4 | Quantum+Super | Quantum+Super | Mamba |
| **4e** | QuantumHydraE2ESuperposition | 4 | Quantum+Super | Quantum+Super | Hydra |

## Key Findings (EEG Ablation Study)

### Best Models by Sampling Frequency

| Frequency | Best Model | Accuracy | Group |
|-----------|-----------|----------|-------|
| 40Hz | 3b TrueClassicalMamba | 73.50 ± 1.35% | Classical |
| **80Hz** | **1c QuantumHydraSSM** | **72.76 ± 1.41%** | **Quantum** |
| 160Hz | 3b TrueClassicalMamba | 72.91 ± 1.05% | Classical |

### Critical Discoveries

1. **Quantum Advantage at 80Hz**: All Group 1 models (quantum features + classical mixing) outperform classical baselines at 80Hz
2. **True Superposition Stability**: Models 2d/2e avoid catastrophic failures seen in 2b/2c at higher frequencies
3. **Hybrid > E2E Quantum**: Quantum feature extraction with classical mixing (Group 1) outperforms end-to-end quantum (Group 4)
4. **Frequency-Dependent Advantage**: Quantum advantage is sampling frequency dependent

### Current Experiment Status

| Experiment | Completed | Total | Status |
|------------|-----------|-------|--------|
| EEG Ablation | 142 | 144 | 98.6% Complete |
| Synthetic Benchmarks | 92 | 396 | ~23% In Progress |

## Directory Structure

```
quantum_hydra_mamba/
├── models/                         # Model implementations (31 files)
│   ├── QuantumTransformer.py       # 1a: Quantum feat → Transformer attention
│   ├── QuantumSSM.py               # 1b/1c: Quantum feat → Classical SSM
│   ├── QuantumHydraSSM.py          # 2d/2e: True superposition mixing
│   ├── QuantumMixingSSM.py         # 2a-2c: Classical feat → Quantum mixing
│   ├── QuantumE2E.py               # 4a-4c: End-to-end quantum
│   ├── QuantumE2E_Superposition.py # 4d/4e: E2E with true superposition
│   ├── ClassicalTransformer.py     # 3a: Classical transformer baseline
│   ├── TrueClassicalMamba.py       # 3b: Classical Mamba baseline
│   ├── TrueClassicalHydra.py       # 3c: Classical Hydra baseline
│   ├── QuantumGatedRecurrence.py   # LSTM-style quantum gating
│   ├── qts_encoder.py              # Shared QTS feature encoder
│   ├── quantum_attention_core.py   # Quantum attention components
│   ├── quantum_hydra_ssm_core.py   # Hydra SSM core (basic)
│   ├── quantum_hydra_ssm_core_advanced.py  # Hydra SSM core (advanced)
│   ├── quantum_mamba_ssm_core.py   # Mamba SSM core (basic)
│   ├── quantum_mamba_ssm_core_advanced.py  # Mamba SSM core (advanced)
│   ├── QTSQuantumTransformer.py    # QTS Transformer wrapper
│   ├── QTSQuantumHydraSSM.py       # QTS Hydra wrapper (basic)
│   ├── QTSQuantumHydraSSMAdvanced.py # QTS Hydra wrapper (advanced)
│   ├── QTSQuantumMambaSSM.py       # QTS Mamba wrapper (basic)
│   ├── QTSQuantumMambaSSMAdvanced.py # QTS Mamba wrapper (advanced)
│   ├── QuantumHydra.py             # Original Hydra model
│   ├── QuantumHydraHybrid.py       # Hybrid Hydra variant
│   ├── QuantumMamba.py             # Original Mamba model
│   ├── QuantumMambaHybrid.py       # Hybrid Mamba variant
│   ├── QuantumMambaLite.py         # Lightweight Mamba
│   ├── QuantumMambaHybridLite.py   # Lightweight hybrid Mamba
│   ├── ClassicalHydra.py           # Classical Hydra (legacy)
│   ├── QuantumAttention.py         # Quantum attention module
│   └── __init__.py                 # Model registry and exports
│
├── scripts/                        # Training and analysis scripts (29 files)
│   ├── run_ablation_eeg.py         # Main EEG ablation training script
│   ├── run_synthetic_benchmark.py  # Synthetic benchmark training script
│   ├── aggregate_ablation_results.py    # EEG ablation results aggregator
│   ├── aggregate_synthetic_results.py   # Synthetic benchmark aggregator
│   ├── generate_ablation_eeg_jobs.py    # EEG job script generator
│   ├── generate_synthetic_jobs.py       # Synthetic job script generator
│   ├── generate_all_synthetic_datasets.py # Dataset generator
│   ├── run_full_comparison.py      # Full model comparison
│   ├── run_gated_models.py         # Gated model training
│   ├── run_gated_genomic.py        # Genomic with gated models
│   ├── run_single_model_*.py       # Single model training scripts
│   ├── aggregate_*_results.py      # Various result aggregators
│   ├── analyze_all_experiments.py  # Cross-experiment analysis
│   ├── evaluate_checkpoint.py      # Checkpoint evaluation
│   ├── QuantumHydraGLUE.py         # GLUE benchmark wrapper
│   ├── run_glue.py                 # GLUE training script
│   ├── run_ssm_genomic_comparison.py    # SSM genomic comparison
│   └── test_*_with_quantum_models.py    # Testing scripts
│
├── data_loaders/                   # Dataset loaders (15 files)
│   ├── Load_PhysioNet_EEG_NoPrompt.py   # PhysioNet EEG (no prompts)
│   ├── Load_FACED_EEG.py           # FACED EEG dataset
│   ├── Load_SEED_EEG.py            # SEED EEG dataset
│   ├── Load_DNA_Sequences.py       # DNA sequence data
│   ├── Load_Genomic_Benchmarks.py  # Genomic benchmarks
│   ├── Load_GLUE.py                # GLUE NLP benchmark
│   ├── Load_Image_Datasets.py      # Image datasets (MNIST, CIFAR, etc.)
│   ├── eeg_datasets.py             # EEG dataset utilities
│   ├── forrelation_dataloader.py   # Forrelation task loader
│   ├── generate_forrelation_dataset.py  # Forrelation generator (v1)
│   ├── generate_forrelation_dataset_v2.py # Forrelation generator (v2)
│   ├── adding_problem_dataloader.py     # Adding problem loader
│   ├── generate_adding_problem.py       # Adding problem generator
│   ├── selective_copy_dataloader.py     # Selective copy loader
│   └── generate_selective_copy.py       # Selective copy generator
│
├── jobs/                           # Ablation study SLURM scripts
│   ├── ablation_eeg/               # EEG ablation jobs (432 scripts)
│   │   ├── logs/                   # SLURM output logs
│   │   └── ablation_*.sh           # Job scripts
│   └── synthetic/                  # Synthetic benchmark jobs (396 scripts)
│       ├── logs/                   # SLURM output logs
│       └── syn_*.sh                # Job scripts
│
├── job_scripts/                    # Original experiment scripts (158 files)
│   ├── eeg/                        # EEG experiment jobs
│   ├── dna/                        # DNA experiment jobs
│   ├── forrelation/                # Forrelation jobs
│   ├── gated/                      # Gated model jobs
│   ├── genomic/                    # Genomic benchmark jobs
│   ├── glue/                       # GLUE benchmark jobs
│   ├── image/                      # Image classification jobs
│   └── *.sh                        # Submission/monitoring scripts
│
├── results/                        # Experiment results
│   ├── ablation_eeg/               # EEG ablation (142/144 results)
│   │   ├── ablation_*_results.json # JSON result files
│   │   ├── ablation_*_model.pt     # Model checkpoints
│   │   └── checkpoints/            # Training checkpoints
│   ├── synthetic_benchmarks/       # Synthetic (92+ results, in progress)
│   │   ├── synthetic_*_results.json
│   │   ├── synthetic_*_model.pt
│   │   └── checkpoints/
│   ├── eeg_results/                # Original EEG results
│   ├── eeg_8q_160hz/               # 8-qubit 160Hz experiments
│   ├── dna_results/                # DNA classification
│   ├── dna_stratified/             # Stratified DNA
│   ├── mnist_results/              # MNIST classification
│   ├── genomic/                    # Genomic benchmarks
│   ├── genomic_ssm_comparison/     # SSM genomic comparison
│   ├── gated/                      # Gated model results
│   ├── glue/                       # GLUE benchmark (9 tasks)
│   ├── forrelation_results/        # Forrelation experiments
│   ├── full_comparison/            # Full model comparison
│   └── multigpu/                   # Multi-GPU experiments
│
├── data/                           # Dataset storage
│   └── synthetic_benchmarks/       # Synthetic benchmark datasets
│       ├── forrelation/            # Forrelation task (L=50,100,200)
│       ├── adding_problem/         # Adding problem (L=100,200,500,1000)
│       └── selective_copy/         # Selective copy (L=100,200,500,1000)
│
├── docs/                           # Documentation (44 markdown files)
│   ├── ABLATION_EEG_RESULTS_SUMMARY.md  # EEG ablation results
│   ├── ABLATION_STUDY_PLAN_V3.md        # Current study plan
│   ├── SYNTHETIC_BENCHMARK_PROTOCOL.md  # Synthetic benchmark protocol
│   ├── GROUP2_PERFORMANCE_ANALYSIS.md   # Group 2 failure analysis
│   ├── RESEARCH_QUESTIONS_ANSWERS.md    # Research findings
│   ├── QUANTUM_SSM_README.md            # SSM documentation
│   ├── QUANTUM_GATED_RECURRENCE_GUIDE.md # Gated model guide
│   ├── Forrelation_Dataset_Usage_Guide.md
│   ├── Forrelation_Experiment_Rationale.md
│   └── ... (40+ additional docs)
│
├── circuits/                       # Quantum circuit diagrams (7 PDFs)
│   ├── 01_qshift_circuit.pdf       # Q-Shift gate
│   ├── 02_qflip_circuit.pdf        # Q-Flip gate
│   ├── 03_qlcu_circuit.pdf         # Linear Combination of Unitaries
│   ├── 04_qd_circuit.pdf           # Q-D gate
│   ├── 05_branch_psi1.pdf          # Branch ψ₁
│   ├── 06_branch_psi2.pdf          # Branch ψ₂
│   └── 07_branch_psi3.pdf          # Branch ψ₃
│
├── Processed_data/                 # Preprocessed datasets
├── download_package/               # Dataset download utilities
├── faced_io_test/                  # FACED dataset I/O tests
├── StudyPlan.md                    # Original study plan
├── requirements_eeg_datasets.txt   # EEG dataset requirements
└── README.md                       # This file
```

## Quick Start

### 1. Environment Setup

```bash
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg
cd /pscratch/sd/j/junghoon/quantum_hydra_mamba
```

### 2. Run Single Ablation Experiment

```bash
python scripts/run_ablation_eeg.py \
    --model-id 1c \
    --sampling-freq 80 \
    --seed 2024 \
    --n-qubits 6 \
    --n-layers 2 \
    --n-epochs 50
```

### 3. Submit SLURM Batch Jobs

```bash
# Single job
sbatch jobs/ablation_eeg/ablation_1c_80Hz_seed2024.sh

# All EEG ablation jobs
cd jobs/ablation_eeg && bash submit_all.sh
```

### 4. Run Synthetic Benchmark

```bash
python scripts/run_synthetic_benchmark.py \
    --model-id 1a \
    --task forrelation \
    --seq-len 100 \
    --seed 2024
```

### 5. Aggregate Results

```bash
# EEG ablation results
python scripts/aggregate_ablation_results.py --input-dir ./results/ablation_eeg

# Synthetic benchmark results
python scripts/aggregate_synthetic_results.py --results-dir ./results/synthetic_benchmarks
```

## Synthetic Benchmark Tasks

Three tasks designed to evaluate quantum advantages for long-range sequence learning:

| Task | Type | Seq Lengths | Purpose |
|------|------|-------------|---------|
| **Forrelation** | Classification | 50, 100, 200 | Quantum superposition advantage test |
| **Adding Problem** | Regression | 100, 200, 500, 1000 | Long-range dependency |
| **Selective Copy** | Multi-output Regression | 100, 200, 500, 1000 | Selective memory |

## Hyperparameters

### Standard Configuration

```python
N_QUBITS = 6
N_LAYERS = 2
D_MODEL = 128
D_STATE = 16
N_EPOCHS = 50 (EEG) / 100 (Synthetic)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING = 10 (EEG) / 20 (Synthetic)
```

### EEG Sampling Frequencies

- 40 Hz (200 timesteps)
- 80 Hz (400 timesteps)
- 160 Hz (800 timesteps)

## Three-Branch Quantum Superposition

All quantum models use three-branch superposition:

```
|ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
```

where α, β, γ are learnable complex coefficients representing:
- ψ₁: Forward temporal processing
- ψ₂: Backward temporal processing
- ψ₃: Global context integration

## Requirements

```
torch>=2.0
pennylane>=0.35
numpy
scikit-learn
tqdm
mne  # For EEG data
datasets>=2.14  # For GLUE
transformers>=4.30  # For GLUE
```

## Monitoring Experiments

```bash
# Check running jobs
squeue -u junghoon | grep -E "ablation|syn_"

# Count completed results
ls results/ablation_eeg/*.json | wc -l
ls results/synthetic_benchmarks/*.json | wc -l

# Watch specific job logs
tail -f jobs/ablation_eeg/logs/ablation_1c_80Hz_seed2024_*.out
```

## Key Documentation

| Document | Description |
|----------|-------------|
| `docs/ABLATION_EEG_RESULTS_SUMMARY.md` | Complete EEG ablation results |
| `docs/ABLATION_STUDY_PLAN_V3.md` | Current experiment plan |
| `docs/SYNTHETIC_BENCHMARK_PROTOCOL.md` | Synthetic benchmark protocol |
| `docs/GROUP2_PERFORMANCE_ANALYSIS.md` | Analysis of Group 2 failures |
| `docs/RESEARCH_QUESTIONS_ANSWERS.md` | Research findings summary |
| `docs/QUANTUM_SSM_README.md` | SSM architecture documentation |

## Related Memory Files

Project memory is maintained in `/pscratch/sd/j/junghoon/mem0-mcp/`:
- `QUANTUM_HYDRA_MAMBA_JANUARY_2026_UPDATE.md` - Latest progress update
- `QUANTUM_HYDRA_MAMBA_MEMORY_SUMMARY.md` - Original project setup
- `SYNTHETIC_LONG_RANGE_EXPERIMENTS.md` - Synthetic benchmark design
- `PROJECT_STATUS_SUMMARY.md` - Overall project tracking

## Author

**Junghoon Park**

---

**Current Status (January 2026):**
- EEG Ablation: 142/144 (98.6%) complete
- Synthetic Benchmarks: 92/396 (~23%) in progress
- Key Finding: Quantum advantage at 80Hz sampling frequency

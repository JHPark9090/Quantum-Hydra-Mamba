# Research Plan

## Ablation Study

### Selective Mechanism
- Gated: LSTM-style gates for selective forgetting (QuantumMambaGated, QuantumHydraGated)
- SSM: State space models with selective scan (QuantumMambaSSM, QuantumHydraSSM)
- Attention: Self-attention for global information mixing (QuantumMambaAttention, QuantumHydraAttention)

#### Dataset
- Use three simplest, smallest tasks from Genomic Benchmarks
    - demo_human_or_worm (200 seq_len)
    - demo_coding_vs_intergenomic (200 seq_len)
    - drosophila_enhancers_stark (smaller)

### Unidirectional vs Bidirectional
- After finding the best selective mechanism model (Gated vs SSM vs Attention), compare model performance between QuantumMamba vs QuantumHydra
- Use results of all the experiments shown below: Image data, EEG, Genomic Benchmarks, GLUE, Forrelation

-----

## Classical Baselines

### True Classical Hydra
- Faithful implementation of Hwang et al. (2024)
- Bidirectional processing
- Semi-separable matrix operations
- Shift, flip, diagonal branches

### True Classical Mamba
- **Architecture:** Faithful implementation of Gu & Dao (2024)
- Selective SSM with input-dependent parameters
- RMSNorm
- Gated MLP blocks

-----

## Experiments

### Image data

**Dataset:** 
- MNIST
- COCO

**Task**: Multi-class classification

**Data Loader**
- Directory: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/data_loaders/Load_Image_Datasets.py`
- This file does not contain COCO dataloader. We have to revise this Python file by writing the correct codes to load COCO dataset.

**Why Essential:**
✅ Standard ML benchmark
✅ Easy comparison with literature
✅ Small enough for quantum encoding
✅ Well-understood baselines

**Aggregation**
- Directory: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/scripts/aggregate_mnist_results.py`
- Compute mean ± std across seeds
- Generates: CSV, LaTeX tables, comparison plots

-----

### Multi-variate Time-series data: PhysioNet EEG

**Dataset:** PhysioNet Motor Imagery

**Data Loader**
- File Directory: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/data_loaders/Load_PhysioNet_EEG_NoPrompt.py`

**Details:**
- **Task:** Binary classification (left/right hand movement)
- **Channels:** 64 EEG channels
- **Sequence Length:** 160 timesteps
- **Classes:** 2 (left hand, right hand)
- **Samples:** 10-20 subjects

**Why Essential:**
✅ Real-world biomedical application
✅ Rich temporal patterns (SSMs excel here)
✅ Manageable scale for quantum circuits
✅ Tests selective SSM mechanisms

**Run experiments:**
```bash
# Single model
python scripts/run_single_model_eeg.py \
    --model-name quantum_hydra \
    --seed 2024 \
    --device cuda

# All 6 models
bash scripts/run_all_eeg_experiments.sh
```

**Aggregation**
- Directory: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/scripts/aggregate_eeg_results.py`
- Compute mean ± std across seeds
- Generates: CSV, LaTeX tables, comparison plots

**Expected Outcome:** Quantum models competitive (75-85% accuracy)

-----

### Genomic Sequence data: Genomic Benchmarks
**Dataset:** Genomic Benchmarks is a collection of curated and easily accessible sequence classification datasets in the field of genomics.
It can be used as a critical standard to compare with existing deep learning based state space models such as HyenaDNA.

**File Directory**
- Data Loader: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/data_loaders/Load_Genomic_Benchmarks.py`

-----

### Language data: GLUE Benchmark (10 tasks)
We adapt the quantum state-space models (QuantumHydra, QuantumMamba) for natural language understanding tasks by adding:
- Token embedding layer with positional encoding
- Sentence pair handling for tasks like MRPC, QQP, RTE
- Task-specific output heads (classification/regression)

**File Directory**
- README file directory: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/docs/GLUE_README.md`
- Data Loader file directory: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/data_loaders/Load_GLUE.py`

-----

### Forrelation (Quantum Advantage Test)

**Dataset:** Sequential Forrelation (generated)

**File Directory**
- Data Generation: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/data_loaders/generate_forrelation_dataset.py`
- Data Loader: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/data_loaders/forrelation_dataloader.py`
- README: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/docs/FORRELATION_README.md`
- Experiment Rationale: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/docs/Forrelation_Experiment_Rationale.md`
- Dataset Usage Guide: `/pscratch/sd/j/junghoon/quantum_hydra_mamba/docs/Forrelation_Dataset_Usage_Guide.md`

**Details:**
- **Task:** Binary classification (high vs low forrelation)
- **Sequence Length:** Variable (20, 40, 80, 160)
- **n_bits:** 6 or 8
- **Purpose:** Test proven quantum advantage (Aaronson & Ambainis, 2015)

**Why CRITICAL:**
✅ Proven quantum advantage exists
✅ Tests sample efficiency (fewer samples needed)
✅ Tests scaling with problem complexity
✅ Designed specifically for quantum computers

**Generate datasets:**
```bash
bash /pscratch/sd/j/junghoon/quantum_hydra_mamba/job_scripts/generate_forrelation_datasets.sh
```

**Run experiments:**
```bash
# All models on all Forrelation datasets
## Must check which of the following two scripts are correct.
bash /pscratch/sd/j/junghoon/quantum_hydra_mamba/job_scripts/run_all_forrelation_experiments.sh
bash /pscratch/sd/j/junghoon/quantum_hydra_mamba/job_scripts/submit_forrelation_experiments.sh

# Analyze quantum advantage
python /pscratch/sd/j/junghoon/quantum_hydra_mamba/scripts/aggregate_forrelation_results.py
```

**Expected Outcome:**
- **Silver Standard**: Quantum models need 30-50% **fewer samples** to reach 90% accuracy
- **Gold Standard**: Advantage **increases** with problem complexity

**Quantum Advantage Evaluation (IMPORTANT!!)**
- README: `pscratch/sd/j/junghoon/quantum_hydra_mamba/docs/Quantum_Advantage_Test_Plan.md`

-----

## Statistical Rigor

**Multiple Runs:**
- Minimum 5 runs per configuration
- Different random seeds: 2024, 2025, 2026, 2027, 2028
- Report mean ± standard deviation

**Statistical Tests:**
- Paired t-test for model comparisons
- Wilcoxon signed-rank test (non-parametric alternative)
- Confidence intervals (95%)
- False Discovery Rate correction for multiple comparisons

**Significance Level:** p < 0.05


-----

## References

1. **Gu, A., & Dao, T. (2024).** Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
   - https://arxiv.org/html/2312.00752v2

2. **Hwang, W., Kim, M., Zhang, X., & Song, H. (2024).** Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers.
   - https://arxiv.org/pdf/2407.09941

3. **Aaronson, S., & Ambainis, A. (2015).** Forrelation: A Problem that Optimally Separates Quantum from Classical Computing.
   - https://arxiv.org/abs/1411.5729


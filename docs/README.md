# Documentation Index

This directory contains comprehensive documentation for the Quantum Hydra and Quantum Mamba models.

## 📚 Main Guides

### 1. [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) ⭐ **START HERE**
**Complete guide for running all experiments**

- Overview of all 6 models (4 quantum + 2 classical)
- Datasets: EEG, MNIST, DNA, Forrelation
- Research questions and hypotheses
- How to run experiments on GPU hardware
- Expected results and analysis

**Use this guide to:**
- Understand the experimental framework
- Run your first experiment
- Interpret results

---

### 2. [QUANTUM_HYDRA_GUIDE.md](QUANTUM_HYDRA_GUIDE.md)
**Detailed guide for Quantum Hydra models**

- Option A (Superposition) vs Option B (Hybrid)
- Mathematical formulations
- Implementation details
- Code examples
- When to use which approach

**Use this guide to:**
- Understand Quantum Hydra architecture
- Choose between Option A and Option B
- Implement custom Quantum Hydra models

---

### 3. [QUANTUM_MAMBA_GUIDE.md](QUANTUM_MAMBA_GUIDE.md)
**Detailed guide for Quantum Mamba models**

- Option A (Superposition) vs Option B (Hybrid)
- Quantum circuit components (SSM, gating, skip)
- Comparison with Quantum Hydra
- Code examples
- Design philosophy

**Use this guide to:**
- Understand Quantum Mamba architecture
- Compare with Quantum Hydra
- Implement custom Quantum Mamba models

---

### 4. [TIMING_AND_METRICS.md](TIMING_AND_METRICS.md) ⭐ **NEW**
**Guide to timing information and performance metrics**

- What timing information is recorded
- Where timing data is saved (JSON files)
- How to access and analyze timing
- Example: Compare training times across models
- Per-epoch timing analysis

**Use this guide to:**
- Find out how long each experiment took
- Compare quantum vs classical training times
- Analyze parameter efficiency
- Extract timing from results files

---

## 🧪 Forrelation Documentation

The following guides explain the Forrelation experiments for testing quantum advantage:

### 4. [FORRELATION_README.md](FORRELATION_README.md)
**Complete Forrelation experiment guide**

- What is Forrelation?
- Why it's ideal for quantum advantage testing
- Dataset generation
- Running experiments
- Analyzing results

---

### 5. [Forrelation_Experiment_Rationale.md](Forrelation_Experiment_Rationale.md)
**Theoretical background for Forrelation**

- Why Forrelation proves quantum advantage
- Sequential Forrelation task design
- Bronze, Silver, Gold standards
- Theoretical expectations

---

### 6. [Forrelation_Dataset_Usage_Guide.md](Forrelation_Dataset_Usage_Guide.md)
**Practical guide for Forrelation datasets**

- How to generate datasets
- Dataset naming conventions
- Loading data with forrelation_dataloader.py
- Troubleshooting

---

### 7. [Quantum_Advantage_Test_Plan.md](Quantum_Advantage_Test_Plan.md)
**Phase-by-phase testing protocol**

- Phase 3: Silver Standard (sample efficiency)
- Phase 4: Gold Standard (scaling with complexity)
- Success criteria
- Statistical analysis

---

### 8. [TIMING_AND_METRICS.md](TIMING_AND_METRICS.md)
**Timing information and performance metrics**

- What timing information is recorded
- How to access timing data
- Analysis examples
- Expected timing results

---

## 🎯 Quick Navigation

### I want to...

**Run my first experiment:**
→ Read [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)
→ Run: `python experiments/run_single_model_eeg.py --model-name quantum_hydra --device cuda`

**Understand Quantum Hydra:**
→ Read [QUANTUM_HYDRA_GUIDE.md](QUANTUM_HYDRA_GUIDE.md)

**Understand Quantum Mamba:**
→ Read [QUANTUM_MAMBA_GUIDE.md](QUANTUM_MAMBA_GUIDE.md)

**Test quantum advantage:**
→ Read [FORRELATION_README.md](FORRELATION_README.md)
→ Run: `bash scripts/generate_forrelation_datasets.sh`
→ Run: `bash scripts/run_all_forrelation_experiments.sh`

**Compare Option A vs Option B:**
→ Read sections in [QUANTUM_HYDRA_GUIDE.md](QUANTUM_HYDRA_GUIDE.md#when-to-use-which)
→ Read sections in [QUANTUM_MAMBA_GUIDE.md](QUANTUM_MAMBA_GUIDE.md#design-philosophy)

**Understand the datasets:**
→ Read [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md#datasets)

**Analyze results:**
→ Read [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md#analysis)
→ Run aggregation scripts in `experiments/`

**Check timing information:**
→ Read [TIMING_AND_METRICS.md](TIMING_AND_METRICS.md)
→ Access timing from JSON results files
→ Compare training times across models

---

## 📖 Reading Order

**For first-time users:**
1. [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) - Overall framework
2. [QUANTUM_HYDRA_GUIDE.md](QUANTUM_HYDRA_GUIDE.md) - Hydra models
3. [QUANTUM_MAMBA_GUIDE.md](QUANTUM_MAMBA_GUIDE.md) - Mamba models
4. [FORRELATION_README.md](FORRELATION_README.md) - Quantum advantage testing

**For researchers:**
1. [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) - Research questions
2. [Forrelation_Experiment_Rationale.md](Forrelation_Experiment_Rationale.md) - Theoretical foundation
3. [Quantum_Advantage_Test_Plan.md](Quantum_Advantage_Test_Plan.md) - Testing protocol
4. Model guides for implementation details

---

## 🔗 External Resources

**Classical Papers:**
- Hwang et al. (2024) - [Hydra: Bidirectional State Space Models](https://arxiv.org/pdf/2407.09941)
- Gu & Dao (2024) - [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/html/2312.00752v2)

**Quantum Advantage:**
- Aaronson & Ambainis (2015) - [Forrelation](https://arxiv.org/abs/1411.5729)

**Framework:**
- [PennyLane Documentation](https://pennylane.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Last Updated:** November 2025
**Maintained by:** Junghoon Park

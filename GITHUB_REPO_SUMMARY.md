# GitHub Repository Summary

**Repository:** Quantum Hydra & Quantum Mamba
**Purpose:** Time-series modeling with quantum circuits
**Target User:** Colleagues running experiments on their own GPU hardware (not SLURM/HPC)

---

## ✅ Repository Contents

### 📂 Directory Structure

```
quantum-hydra-mamba/
├── models/                          # Model implementations (6 models)
│   ├── QuantumHydra.py             # Quantum Hydra (superposition)
│   ├── QuantumHydraHybrid.py       # Quantum Hydra (hybrid)
│   ├── QuantumMamba.py             # Quantum Mamba (superposition)
│   ├── QuantumMambaHybrid.py       # Quantum Mamba (hybrid)
│   ├── TrueClassicalHydra.py       # Classical Hydra baseline
│   └── TrueClassicalMamba.py       # Classical Mamba baseline
│
├── datasets/                        # Data loading utilities
│   ├── Load_Image_Datasets.py      # MNIST, Fashion-MNIST loaders
│   ├── Load_PhysioNet_EEG_NoPrompt.py  # EEG data loader
│   ├── Load_DNA_Sequences.py       # DNA promoter data loader
│   ├── generate_forrelation_dataset.py  # Forrelation dataset generator
│   └── forrelation_dataloader.py   # Forrelation data loader
│
├── experiments/                     # Training scripts
│   ├── run_single_model_eeg.py     # EEG experiments
│   ├── run_single_model_mnist.py   # MNIST experiments
│   ├── run_single_model_dna.py     # DNA experiments
│   ├── run_single_model_forrelation.py  # Forrelation experiments
│   ├── aggregate_eeg_results.py    # Aggregate EEG results
│   ├── aggregate_mnist_results.py  # Aggregate MNIST results
│   ├── aggregate_dna_results.py    # Aggregate DNA results
│   └── aggregate_forrelation_results.py  # Analyze quantum advantage
│
├── scripts/                         # Convenience scripts
│   ├── run_all_eeg_experiments.sh
│   ├── run_all_mnist_experiments.sh
│   ├── run_all_dna_experiments.sh
│   ├── generate_forrelation_datasets.sh
│   └── run_all_forrelation_experiments.sh
│
├── docs/                            # Documentation
│   ├── README.md                   # Documentation index ⭐ START HERE
│   ├── EXPERIMENT_GUIDE.md         # Main experiment guide
│   ├── QUANTUM_HYDRA_GUIDE.md      # Quantum Hydra documentation
│   ├── QUANTUM_MAMBA_GUIDE.md      # Quantum Mamba documentation
│   ├── FORRELATION_README.md       # Forrelation experiments
│   ├── Forrelation_Experiment_Rationale.md
│   ├── Forrelation_Dataset_Usage_Guide.md
│   └── Quantum_Advantage_Test_Plan.md
│
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore file
└── README.md                        # Main repository README
```

---

## 🎯 Quick Start for Your Colleague

> **⚡ SUPER QUICK:** See `QUICK_START.md` for 30-second guide with just commands!

### 1. Clone Repository
```bash
git clone <your-github-url>
cd quantum-hydra-mamba
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run First Experiment
```bash
# Test a single model on EEG data
python experiments/run_single_model_eeg.py \
    --model-name quantum_hydra \
    --n-qubits 6 \
    --n-epochs 50 \
    --batch-size 32 \
    --seed 2024 \
    --device cuda
```

### 4. Run All Experiments
```bash
# Run all 6 models on EEG (18 experiments: 6 models × 3 seeds)
bash scripts/run_all_eeg_experiments.sh

# Run all 6 models on MNIST
bash scripts/run_all_mnist_experiments.sh

# Run all 6 models on DNA
bash scripts/run_all_dna_experiments.sh

# Generate Forrelation datasets
bash scripts/generate_forrelation_datasets.sh

# Run Forrelation experiments (144 total)
bash scripts/run_all_forrelation_experiments.sh
```

---

## 📚 Documentation

**All documentation is in the `docs/` folder.**

**Start here:** [docs/README.md](docs/README.md)

This index explains:
- Which guide to read for which purpose
- Quick navigation for common tasks
- Reading order for first-time users

---

## 🧬 The Six Models

| Model | File | Type | Description |
|-------|------|------|-------------|
| **Quantum Hydra (Superposition)** | `models/QuantumHydra.py` | Quantum A | α\|ψ₁⟩ + β\|ψ₂⟩ + γ\|ψ₃⟩ |
| **Quantum Hydra (Hybrid)** | `models/QuantumHydraHybrid.py` | Quantum B | w₁·y₁ + w₂·y₂ + w₃·y₃ |
| **Quantum Mamba (Superposition)** | `models/QuantumMamba.py` | Quantum A | α\|ψ_ssm⟩ + β\|ψ_gate⟩ + γ\|ψ_skip⟩ |
| **Quantum Mamba (Hybrid)** | `models/QuantumMambaHybrid.py` | Quantum B | w₁·y_ssm + w₂·y_gate + w₃·y_skip |
| **True Classical Hydra** | `models/TrueClassicalHydra.py` | Classical | Hwang et al. (2024) baseline |
| **True Classical Mamba** | `models/TrueClassicalMamba.py` | Classical | Gu & Dao (2024) baseline |

**Total:** 4 quantum models + 2 classical baselines = 6 models

---

## 📊 Datasets

1. **PhysioNet EEG** - Motor imagery classification (64 channels, 160 timesteps)
2. **MNIST** - Image classification (28×28, 10 classes)
3. **DNA Promoter** - Sequence classification (57 nucleotides, binary)
4. **Forrelation** - Quantum advantage test (varying n_bits and seq_len)

---

## 🎯 Research Questions

1. **Do quantum models outperform classical baselines?**
   - Compare accuracy across all 6 models
   - Test on 4 different datasets

2. **Is quantum advantage architecture-specific?**
   - Quantum Hydra vs Quantum Mamba
   - Which architecture benefits more from quantization?

3. **Does quantum superposition help?**
   - Superposition (Option A) vs Hybrid (Option B)
   - When does quantum interference provide benefits?

4. **Where do quantum models excel?**
   - EEG, MNIST, DNA, or Forrelation?
   - Task characteristics that favor quantum models

5. **Is there genuine quantum advantage?**
   - Forrelation experiments test this rigorously
   - Sample efficiency and scaling analysis

---

## 📈 Expected Performance

| Dataset | Quantum Models | Classical Models | Gap |
|---------|----------------|------------------|-----|
| **EEG** | 75-85% | 80-85% | Competitive |
| **MNIST** | 85-90% | 95-98% | Classical better |
| **DNA** | 70-80% | 65-75% | Quantum better? |
| **Forrelation** | High with fewer samples | High with more samples | **Quantum advantage** |

---

## 🛠️ Hardware Requirements

### Minimum
- **GPU**: NVIDIA GPU with 4GB+ VRAM (e.g., GTX 1650)
- **RAM**: 8GB
- **Storage**: 5GB

### Recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3060)
- **RAM**: 16GB
- **Storage**: 10GB

### Running on CPU
All scripts support CPU execution (remove `--device cuda` flag), but will be **much slower** (~10-20× slower).

---

## 📝 Key Features

### ✅ All Quantum & Classical Models
- 4 quantum models (2 architectures × 2 design approaches)
- 2 classical baselines for fair comparison
- All models tested and working

### ✅ Complete Dataset Loaders
- PhysioNet EEG (automatic download via MNE)
- MNIST (automatic download via PyTorch)
- DNA Sequences (automatic download via UCI ML)
- Forrelation (generate with provided script)

### ✅ Experiment Scripts
- Single model training scripts for each dataset
- Batch scripts to run all 6 models
- Aggregation scripts for result analysis

### ✅ Documentation
- Main README with quick start guide
- Detailed model guides (Quantum Hydra & Mamba)
- Experiment guide with research questions
- Forrelation documentation for quantum advantage testing

### ✅ No SLURM/HPC Dependencies
- All scripts run on standard GPU hardware
- Simple `bash` scripts for batch execution
- No job submission or cluster management

---

## 🚀 What Your Colleague Can Do

### Immediate (Day 1)
1. Clone repository
2. Install dependencies
3. Run first experiment on EEG data
4. Verify GPU setup works

### Short-term (Week 1)
1. Run all experiments on EEG dataset
2. Run all experiments on MNIST dataset
3. Generate and run Forrelation experiments
4. Compare quantum vs classical performance

### Long-term (Month 1)
1. Complete all 4 datasets
2. Analyze results across all models
3. Generate figures for publication
4. Write up findings

---

## 📧 Next Steps

**For your colleague:**
1. **Read:** [docs/README.md](docs/README.md) (Documentation index)
2. **Read:** [docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md) (Main experiment guide)
3. **Install:** `pip install -r requirements.txt`
4. **Run:** First experiment following Quick Start above
5. **Analyze:** Results using aggregation scripts

**For you:**
1. Push repository to GitHub
2. Share repository URL with colleague
3. Ensure they have GPU access
4. Support as needed

---

## ✅ Repository Checklist

- ✅ All 6 model files (4 quantum + 2 classical)
- ✅ All 5 dataset loaders (EEG, MNIST, DNA, Forrelation generator + loader)
- ✅ All 4 experiment scripts (run_single_model_*.py)
- ✅ All 4 aggregation scripts (aggregate_*_results.py)
- ✅ All 5 batch scripts (run_all_*_experiments.sh)
- ✅ Complete documentation (8 markdown files in docs/)
- ✅ Main README with overview
- ✅ requirements.txt with all dependencies
- ✅ .gitignore for proper version control

**Status:** ✅ **READY FOR DISTRIBUTION**

---

## 📚 Additional Resources

**Classical Papers:**
- Hwang et al. (2024) - https://arxiv.org/pdf/2407.09941
- Gu & Dao (2024) - https://arxiv.org/html/2312.00752v2

**Quantum Advantage:**
- Aaronson & Ambainis (2015) - https://arxiv.org/abs/1411.5729

**Frameworks:**
- PennyLane - https://pennylane.ai/
- PyTorch - https://pytorch.org/

---

**Last Updated:** November 2025
**Status:** Production-ready
**Purpose:** Colleague can run experiments on own GPU hardware

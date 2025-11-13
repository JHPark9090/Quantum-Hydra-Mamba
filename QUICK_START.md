# Quick Start - 30 Second Guide

Run all 6 models (4 quantum + 2 classical) on all 4 datasets in 5 simple steps.

---

## Step 1: Setup Dependencies (one time)

```bash
pip install -r requirements.txt
```

---

## Step 2: EEG Experiments (18 runs: 6 models × 3 seeds)

```bash
bash scripts/run_all_eeg_experiments.sh
```

**Or run single model:**
```bash
python experiments/run_single_model_eeg.py --model-name quantum_hydra --seed 2024 --device cuda
```

**Results:** `eeg_results/`

---

## Step 3: MNIST Experiments (18 runs: 6 models × 3 seeds)

```bash
bash scripts/run_all_mnist_experiments.sh
```

**Or run single model:**
```bash
python experiments/run_single_model_mnist.py --model-name quantum_mamba --seed 2024 --device cuda
```

**Results:** `mnist_results/`

---

## Step 4: DNA Experiments (18 runs: 6 models × 3 seeds)

```bash
bash scripts/run_all_dna_experiments.sh
```

**Or run single model:**
```bash
python experiments/run_single_model_dna.py --model-name classical_hydra --seed 2024 --device cuda
```

**Results:** `dna_results/`

---

## Step 5: Forrelation Experiments (144 runs: 6 models × 8 datasets × 3 seeds)

**Step 5a: Generate datasets first**
```bash
bash scripts/generate_forrelation_datasets.sh
```

**Step 5b: Run all experiments**
```bash
bash scripts/run_all_forrelation_experiments.sh
```

**Or run single model on one dataset:**
```bash
python experiments/run_single_model_forrelation.py \
    --model-name quantum_hydra \
    --dataset-path forrelation_data/forrelation_L20.pt \
    --seed 2024 \
    --device cuda
```

**Results:** `forrelation_results/`

---

## Step 6: Get Results Summary (performance, timing, etc.)

```bash
# EEG summary
python experiments/aggregate_eeg_results.py

# MNIST summary
python experiments/aggregate_mnist_results.py

# DNA summary
python experiments/aggregate_dna_results.py

# Forrelation summary (quantum advantage analysis)
python experiments/aggregate_forrelation_results.py
```

**Output files:**
- `eeg_results/eeg_all_results.csv`
- `mnist_results/mnist_all_results.csv`
- `dna_results/dna_all_results.csv`
- `forrelation_results/forrelation_all_results.csv`
- `forrelation_results/forrelation_sample_efficiency.png`
- `forrelation_results/forrelation_heatmap.png`

---

## Model Names

Use these with `--model-name`:

- `quantum_hydra` - Quantum Hydra (superposition)
- `quantum_hydra_hybrid` - Quantum Hydra (hybrid)
- `quantum_mamba` - Quantum Mamba (superposition)
- `quantum_mamba_hybrid` - Quantum Mamba (hybrid)
- `classical_hydra` - Classical Hydra baseline
- `classical_mamba` - Classical Mamba baseline

---

## Run on CPU (if no GPU)

Remove `--device cuda` from commands:
```bash
python experiments/run_single_model_eeg.py --model-name quantum_hydra --seed 2024
```

⚠️ **Warning:** CPU is ~10-20× slower than GPU.

---

## Estimated Time (with GPU)

- **EEG**: ~15-18 hours (18 experiments × ~50 min each)
- **MNIST**: ~10-12 hours (18 experiments × ~35 min each)
- **DNA**: ~6-8 hours (18 experiments × ~25 min each)
- **Forrelation**: ~70-80 hours (144 experiments × ~30 min each)

**Total:** ~100-120 hours for all experiments

**Tip:** Run in `screen` or `tmux` to avoid interruption.

---

**That's it!** See `README.md` for detailed documentation.

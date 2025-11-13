# Quantum Mamba Models - Comprehensive Guide

> Quantum implementations of the Mamba architecture for ablation studies

## 🎯 Overview

This document describes **two quantum implementations of the Mamba architecture** (Gu & Dao, 2024) designed for ablation studies comparing quantum and classical state-space models.

### Why Quantum Mamba?

1. **Ablation Study**: Compare Quantum Hydra vs Quantum Mamba to determine if quantum advantages are architecture-specific
2. **Mamba Architecture**: State-of-the-art selective SSM with input-dependent parameters
3. **Two Design Options**: Test both superposition (Option A) and hybrid (Option B) approaches

---

## 📊 The Two Quantum Mamba Models

### **Option A: Quantum Mamba (Superposition)** 🔵

**File:** `QuantumMamba.py`

**Mathematical Formulation:**
```
|ψ⟩ = α|ψ_ssm⟩ + β|ψ_gate⟩ + γ|ψ_skip⟩
where α, β, γ ∈ ℂ (complex coefficients)

|ψ_ssm⟩  = Q_SelectiveSSM|X⟩    (state-space path)
|ψ_gate⟩ = Q_Gating|X⟩          (gating path)
|ψ_skip⟩ = Q_Skip|X⟩            (skip connection)
```

**Key Features:**
- **Quantum superposition** of three pathways before measurement
- **Complex-valued trainable coefficients** (α, β, γ)
- **Single measurement** on combined quantum state
- **Potential for quantum interference** between SSM, gating, and skip paths
- **Fewer quantum circuit calls** per forward pass

**Advantages:**
- ✓ Quantum interference may capture non-classical correlations
- ✓ True quantum superposition (not just classical mixing)
- ✓ Exponential state space (2^n)
- ✓ Fewer measurements required

**Limitations:**
- ✗ Different semantics from classical Mamba
- ✗ Sensitive to quantum decoherence
- ✗ Complex coefficient optimization
- ✗ Harder to interpret

---

### **Option B: Quantum Mamba (Hybrid)** 🟢

**File:** `QuantumMambaHybrid.py`

**Mathematical Formulation:**
```
y₁ = Measure(Q_SelectiveSSM|X⟩)
y₂ = Measure(Q_Gating|X⟩)
y₃ = Measure(Q_Skip|X⟩)

Y = OutputLayer(w₁·y₁ + w₂·y₂ + w₃·y₃)
where w₁, w₂, w₃ ∈ ℝ (real weights)
```

**Key Features:**
- **Three independent quantum circuits**
- **Classical weighted combination** of measurements
- **Real-valued trainable weights**
- **Faithful to classical Mamba's addition semantics**
- **More interpretable** branch contributions

**Advantages:**
- ✓ Preserves classical Mamba semantics
- ✓ Interpretable (can analyze each branch)
- ✓ More robust to quantum noise
- ✓ Real-valued weights (easier optimization)
- ✓ Branch contribution analysis available

**Limitations:**
- ✗ No quantum interference
- ✗ More quantum circuit calls (three separate circuits)
- ✗ Higher computational cost

---

## 🔬 Quantum Circuit Components

### 1. Quantum Selective SSM Circuit

Implements the selective state-space model with **input-dependent B, C, dt parameters**.

```python
def selective_ssm_circuit(qlcu_params, b_params, c_params, dt_params):
    """
    Quantum implementation of selective SSM.

    Classical Mamba SSM:
        h[t] = A * h[t-1] + B(u) * u[t]
        y[t] = C(u) * h[t] + D * u[t]

    Quantum implementation:
        - QLCU: Simulates A matrix (state evolution)
        - b_params: Input-dependent B matrix
        - c_params: Input-dependent C matrix
        - dt_params: Time-step parameters
    """
    # A matrix transformation (via QLCU)
    for layer in range(qlcu_layers):
        qml.RY(qlcu_params)
        qml.IsingXX(qlcu_params)

        # Input-dependent B matrix
        qml.RY(b_params)

        qml.RY(qlcu_params)
        qml.IsingYY(qlcu_params)

        # Output-dependent C matrix
        qml.RZ(c_params)

        # Time-step modulation
        qml.RX(dt_params)
```

**Key features:**
- ✓ Input-dependent parameters (B, C, dt)
- ✓ QLCU for state transformation
- ✓ Time-step modulation
- ✓ Entanglement for state mixing

---

### 2. Quantum Gating Circuit

Implements the multiplicative gating mechanism (like SiLU gating in Mamba).

```python
def gating_circuit(gate_params):
    """
    Quantum gating mechanism.

    Classical Mamba:
        output = y * silu(z)

    Quantum implementation:
        Controlled gates for multiplicative gating
    """
    for layer in range(gate_layers):
        qml.RX(gate_params)
        qml.CRZ(gate_params)  # Gating via controlled rotations
        qml.RY(gate_params)
        qml.CRZ(gate_params)
```

**Key features:**
- ✓ Controlled gates for gating
- ✓ Ring connectivity
- ✓ Mimics SiLU gating behavior

---

### 3. Quantum Skip Circuit

Implements skip connection (D matrix in Mamba).

```python
def skip_circuit(skip_params):
    """
    Quantum skip connection.

    Classical Mamba:
        y[t] += D * u[t]

    Quantum implementation:
        Diagonal operations for direct path
    """
    qml.RX(skip_params)
    qml.RY(skip_params)
    qml.RZ(skip_params)
```

**Key features:**
- ✓ Diagonal operations
- ✓ Independent on each qubit
- ✓ Direct input-to-output path

---

## 🚀 Usage

### Quick Start

```python
import torch
from QuantumMamba import QuantumMambaTS
from QuantumMambaHybrid import QuantumMambaHybridTS

# Option A: Superposition
model_a = QuantumMambaTS(
    n_qubits=6,
    n_timesteps=160,
    qlcu_layers=2,
    feature_dim=64,
    output_dim=2,
    device="cuda"
)

# Option B: Hybrid
model_b = QuantumMambaHybridTS(
    n_qubits=6,
    n_timesteps=160,
    qlcu_layers=2,
    feature_dim=64,
    output_dim=2,
    device="cuda"
)

# Forward pass
x = torch.randn(16, 64, 160)  # (batch, channels, timesteps)
output_a = model_a(x)  # (16, 2)
output_b = model_b(x)  # (16, 2)
```

### Branch Contribution Analysis (Option B only)

```python
# Analyze which branch contributes most
contributions = model_b.quantum_mamba.get_branch_contributions(x[:4])

print(f"SSM branch shape: {contributions['branch1_ssm'].shape}")
print(f"Gate branch shape: {contributions['branch2_gate'].shape}")
print(f"Skip branch shape: {contributions['branch3_skip'].shape}")
print(f"Weights: SSM={contributions['weights']['w1_ssm']:.3f}, "
      f"Gate={contributions['weights']['w2_gate']:.3f}, "
      f"Skip={contributions['weights']['w3_skip']:.3f}")
```

### Training Example

```python
import torch.nn as nn
import torch.optim as optim

# Setup
model = QuantumMambaTS(n_qubits=6, output_dim=2, device="cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to("cuda")
        y_batch = y_batch.to("cuda")

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

---

## 🔍 Comparison with Quantum Hydra

| Aspect | Quantum Hydra | Quantum Mamba |
|--------|---------------|---------------|
| **Base Architecture** | Bidirectional Hydra (shift/flip) | Selective SSM (input-dependent) |
| **Key Mechanism** | Shift, Flip, Diagonal | Selective B/C/dt, Gating, Skip |
| **Classical Paper** | Hwang et al. 2024 | Gu & Dao 2024 |
| **Branch 1** | Qshift(QLCU\|X⟩) | Q_SelectiveSSM\|X⟩ |
| **Branch 2** | Qflip(Qshift(QLCU(Qflip\|X⟩))) | Q_Gating\|X⟩ |
| **Branch 3** | QD\|X⟩ | Q_Skip\|X⟩ |
| **Superposition** | α\|ψ₁⟩ + β\|ψ₂⟩ + γ\|ψ₃⟩ | α\|ψ_ssm⟩ + β\|ψ_gate⟩ + γ\|ψ_skip⟩ |
| **Hybrid** | w₁·y₁ + w₂·y₂ + w₃·y₃ | w₁·y₁ + w₂·y₂ + w₃·y₃ |

---

## 📝 Key Differences from Classical Mamba

### Classical Mamba (Gu & Dao, 2024)

```python
# Selective SSM
B = Linear(input)     # Input-dependent
C = Linear(input)     # Input-dependent
dt = Softplus(Linear(input))

# State evolution
h[t] = A * h[t-1] + B * u[t]
y[t] = C * h[t] + D * u[t]

# Gating
output = y * silu(gate)
```

### Quantum Mamba (This Implementation)

```python
# Quantum selective SSM
b_params = f(input)   # Input-dependent quantum params
c_params = f(input)
dt_params = f(input)

# Quantum state evolution (via QLCU)
|ψ[t]⟩ = QLCU(b, c, dt) |ψ[t-1]⟩

# Three quantum paths
|ψ_ssm⟩ = Q_SelectiveSSM(b, c, dt) |X⟩
|ψ_gate⟩ = Q_Gating |X⟩
|ψ_skip⟩ = Q_Skip |X⟩

# Superposition (Option A) or Classical (Option B)
|ψ⟩ = α|ψ_ssm⟩ + β|ψ_gate⟩ + γ|ψ_skip⟩  (A)
y = w₁·y₁ + w₂·y₂ + w₃·y₃                (B)
```

---

## 💡 Design Philosophy

### Why Three Branches?

Both Quantum Hydra and Quantum Mamba use three branches following their classical counterparts:

**Classical Hydra:**
```
QS(X) = shift(SS(X)) + flip(shift(SS(flip(X)))) + DX
        └─ Branch 1─┘   └────── Branch 2 ──────┘   └─3─┘
```

**Classical Mamba:**
```
output = SSM(u) + Gate(u) + Skip(u)
         └─ 1 ─┘   └─ 2 ─┘   └─ 3 ─┘
```

**Quantum versions maintain this structure** to enable fair comparison with classical baselines.

### Why Two Options (A & B)?

- **Option A (Superposition)**: Tests if quantum interference helps
- **Option B (Hybrid)**: More faithful to classical semantics, easier to interpret

By implementing both, we can determine:
1. If quantum advantages exist
2. If they come from interference (A) or quantum circuits alone (B)

---

## 📚 References

### Papers

1. **Gu & Dao (2024)** - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
   - https://arxiv.org/html/2312.00752v2
   - Original Mamba architecture

2. **Hwang et al. (2024)** - "Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers"
   - https://arxiv.org/pdf/2407.09941
   - Hydra architecture for comparison

---

**Last Updated:** November 2024
**Status:** Production-ready
**Purpose:** Ablation study for quantum vs classical SSMs

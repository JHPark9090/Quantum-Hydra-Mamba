## TrueClassicalHydra

This document describes the reference Hydra-style implementation in
`models/mandy_group2/TrueClassicalHydra.py`.

### Purpose

`TrueClassicalHydra` is a **reference** (non-fused) implementation that follows
the official Hydra execution path as closely as possible in pure PyTorch. It is
meant for correctness checks and small/medium workloads, not for speed.

### Why We Verify Against `models//TrueClassicalHydra.py`

The ablation version in `models/TrueClassicalHydra.py` is an
earlier, simplified Hydra-like model that:

- Uses a different projection split and SSM parameterization
- Applies convolution only to the `x` branch
- Uses a simplified SSM recurrence and gating

We verify against it to:

1. **Confirm behavior differences**: ensure the updated reference code produces
   expected shape/flow changes relative to the simplified ablation baseline.
2. **Track regressions**: if results diverge too far, it helps diagnose whether
   the change is from the new Hydra path or from unrelated refactors.
3. **Preserve historical baselines**: prior experiments used the ablation model;
   keeping it verifiable ensures comparisons remain interpretable.

### What This Implementation Matches

The `HydraBlock` in `TrueClassicalHydra.py` mirrors the official logic:

- Head/grouped parameterization (`headdim`, `ngroups`)
- `xBC` depthwise convolution with centered padding
- Bidirectional scan via forward + flipped backward concatenation
- `dt_bias` and `dt_limit` handling
- Gated RMSNorm (norm-before-gate)
- Diagonal/skip term using headwise `D` and `fc_D`

The scan itself uses a Python loop (`hydra_chunk_scan_reference`) to emulate
the fused Triton kernel.

### Constraints and Notes

- `d_conv` must be **odd** to preserve sequence length with centered padding.
- `expand * d_model` must be divisible by `headdim`.
- `nheads` must be divisible by `ngroups`.
- Performance is slow for long sequences; this is a correctness reference.

### How to Use

Basic example for sequence classification:

```python
import torch
from models.mandy_group2.TrueClassicalHydra import TrueClassicalHydra

batch_size = 4
n_channels = 64
n_timesteps = 160

model = TrueClassicalHydra(
    n_channels=n_channels,
    n_timesteps=n_timesteps,
    d_model=128,
    d_state=16,
    n_layers=2,
    d_conv=7,
    expand=2,
    headdim=64,
    ngroups=1,
    output_dim=2,
    dropout=0.1,
    device="cpu",
)

x = torch.randn(batch_size, n_channels, n_timesteps)
logits = model(x)
print(logits.shape)  # (4, 2)
```

### Related Files

- `models/TrueClassicalHydra.py`
- `models/TrueClassicalHydra.py`

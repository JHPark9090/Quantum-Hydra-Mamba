import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
True Classical Hydra Implementation

Based on "Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers"
by Hwang, Lahoti, Dao, and Gu (arXiv:2407.09941)

Key components:
1. State Space Model (SSM) foundation with A, B, C, D matrices
2. Quasiseparable matrix structure for efficient bidirectional processing
3. Selective state space mechanism (like Mamba)
4. True bidirectional processing via forward + flipped backward passes
5. Time-step modulation (dt parameters)

Reference: https://github.com/goombalab/hydra
"""


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.weight * x_normalized

class RMSNormGated(nn.Module):
    """
    RMSNorm with gated output, matching Hydra's norm-before-gate behavior.
    """

    def __init__(self, d_model, eps=1e-5, norm_before_gate=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.norm_before_gate = norm_before_gate

    def forward(self, x, gate):
        if self.norm_before_gate:
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            x = self.weight * (x / rms)
            return x * F.silu(gate)
        x = x * F.silu(gate)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)

def hydra_chunk_scan_reference(x, dt, A, B, C, initial_states=None, dt_limit=(0.0, float("inf"))):
    """
    Reference (slow) Hydra scan matching mamba_chunk_scan_combined semantics.

    Args:
        x: (B2, L, H, P)
        dt: (B2, L, H)
        A: (H,)
        B: (B2, L, G, N)
        C: (B2, L, G, N)
        initial_states: optional (B2, H, P, N) or (H, P, N)
        dt_limit: tuple (min, max) for clamping dt

    Returns:
        y: (B2, L, H, P)
    """
    b2, seqlen, nheads, headdim = x.shape
    _, _, ngroups, d_state = B.shape
    if nheads % ngroups != 0:
        raise ValueError(f"nheads ({nheads}) must be divisible by ngroups ({ngroups})")

    heads_per_group = nheads // ngroups
    dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])

    if initial_states is None:
        state = torch.zeros(b2, nheads, headdim, d_state, device=x.device, dtype=x.dtype)
    else:
        if initial_states.dim() == 3:
            state = initial_states.unsqueeze(0).expand(b2, -1, -1, -1).contiguous()
        else:
            state = initial_states.contiguous()

    outputs = []
    for t in range(seqlen):
        x_t = x[:, t, :, :]  # (B2, H, P)
        dt_t = dt[:, t, :]  # (B2, H)
        B_t = B[:, t, :, :]  # (B2, G, N)
        C_t = C[:, t, :, :]  # (B2, G, N)

        B_head = B_t.repeat_interleave(heads_per_group, dim=1)  # (B2, H, N)
        C_head = C_t.repeat_interleave(heads_per_group, dim=1)  # (B2, H, N)

        exp_term = torch.exp(A[None, :, None, None] * dt_t[:, :, None, None])
        state = state * exp_term + x_t.unsqueeze(-1) * B_head.unsqueeze(2)
        y_t = torch.sum(state * C_head.unsqueeze(2), dim=-1)  # (B2, H, P)
        outputs.append(y_t)

    return torch.stack(outputs, dim=1)

def shift_right(y):
    # y: (B, L, d)
    y = torch.roll(y, shifts=1, dims=1)
    y[:, 0, :] = 0.0
    return y

class HydraBlock(nn.Module):
    """
    Hydra block implementing the official Hydra execution path.

    Architecture:
    1. Input projection to expand dimension
    2. Split into gate (z) and value (x) branches
    3. 1D convolution for local context
    4. Bidirectional SSM via forward + flipped backward processing
    5. Gated output with RMSNorm

    Args:
        d_model: Model dimension
        d_state: State space dimension
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        dt_rank: Rank for dt projection
    """

    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=7,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        if self.d_conv % 2 == 0:
            raise ValueError("d_conv must be odd to preserve sequence length with centered padding")
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.headdim = headdim
        self.ngroups = ngroups
        if self.d_inner % self.headdim != 0:
            raise ValueError(f"d_inner ({self.d_inner}) must be divisible by headdim ({self.headdim})")
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * (2 * self.ngroups * self.d_state) + 2 * self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias)

        conv_dim = self.d_inner + 2 * (2 * self.ngroups * self.d_state)
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv // 2,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state))

        self.act = nn.SiLU()

        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        A = torch.ones(self.nheads, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.nheads))
        self.fc_D = nn.Linear(self.d_inner, self.nheads, bias=False)

        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, u):
        """
        Bidirectional Hydra forward pass (reference implementation).

        Args:
            u: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log.float())
        initial_states = None
        if self.learnable_init_states:
            initial_states = self.init_states.unsqueeze(0).expand(2 * batch, -1, -1, -1).contiguous()

        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * (2 * self.ngroups * self.d_state), 2 * self.nheads],
            dim=-1
        )

        dt = torch.cat((dt[:, :, :self.nheads], torch.flip(dt[:, :, self.nheads:], (1,))), dim=0)
        dt = F.softplus(dt + self.dt_bias)

        xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
        x, BC = torch.split(
            xBC,
            [self.d_inner, 2 * (2 * self.ngroups * self.d_state)],
            dim=-1
        )
        x_og = x
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)
        BC = torch.cat(
            (BC[:, :, :2 * self.ngroups * self.d_state],
             torch.flip(BC[:, :, 2 * self.ngroups * self.d_state:], (1,))),
            dim=0
        )
        B, C = torch.split(BC, [self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        x_scan = x.view(2 * batch, seq_len, self.nheads, self.headdim)
        B_scan = B.view(2 * batch, seq_len, self.ngroups, self.d_state)
        C_scan = C.view(2 * batch, seq_len, self.ngroups, self.d_state)

        y = hydra_chunk_scan_reference(
            x_scan,
            dt,
            A,
            B_scan,
            C_scan,
            initial_states=initial_states,
            dt_limit=self.dt_limit,
        )

        y = y.reshape(2 * batch, seq_len, self.d_inner)
        y = shift_right(y)
        y_fw, y_bw = y[:batch], torch.flip(y[batch:], (1,))

        diag_heads = F.linear(x_og, self.fc_D.weight, bias=self.D)  # (B, L, H)
        diag = diag_heads.unsqueeze(-1).expand(batch, seq_len, self.nheads, self.headdim)
        diag = diag.reshape(batch, seq_len, self.d_inner)
        y = y_fw + y_bw + x_og * diag

        y = self.norm(y, z)
        return self.out_proj(y)


class TrueClassicalHydra(nn.Module):
    """
    Complete Classical Hydra model for sequence classification.

    True implementation based on the official Hydra paper and codebase.

    Architecture:
        Input -> Embedding -> Hydra Blocks -> Pooling -> Classifier

    Args:
        n_channels: Number of input channels (e.g., 64 for EEG)
        n_timesteps: Sequence length (e.g., 160)
        d_model: Model dimension
        d_state: State space dimension
        n_layers: Number of Hydra blocks
        d_conv: Convolution kernel size
        expand: Expansion factor
        output_dim: Number of output classes
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_channels,
        n_timesteps,
        d_model=128,
        d_state=16,
        n_layers=2,
        d_conv=7,
        expand=2,
        headdim=64,
        ngroups=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=False,
        output_dim=2,
        dropout=0.1,
        device: str = "cpu",
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.d_model = d_model
        self.device = device

        # Input embedding: project channels to d_model
        self.embedding = nn.Linear(n_channels, d_model)
        self.dropout = nn.Dropout(dropout)

        # Stack of Hydra blocks
        self.layers = nn.ModuleList([
            HydraBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                ngroups=ngroups,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
                dt_limit=dt_limit,
                learnable_init_states=learnable_init_states,
                activation=activation,
                bias=bias,
                conv_bias=conv_bias,
                chunk_size=chunk_size,
                use_mem_eff_path=use_mem_eff_path,
            )
            for _ in range(n_layers)
        ])

        # Layer normalization after each block
        self.norms = nn.ModuleList([
            RMSNorm(d_model) for _ in range(n_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(d_model)

        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        # Move all parameters to specified device
        self.to(device)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, n_channels, n_timesteps) for 3D inputs (e.g., EEG)
               OR (batch, features) for 2D inputs (e.g., DNA, MNIST)

        Returns:
            output: (batch, output_dim)
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # 2D input: (B, features) -> (B, features, 1) so transpose yields (B, 1, features)
            if x.shape[1] != self.n_channels:
                raise ValueError(
                    f"2D input features ({x.shape[1]}) must match n_channels ({self.n_channels})"
                )
            x = x.unsqueeze(-1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D tensor")

        # Now x is 3D: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)

        # Embed: (B, T, C) -> (B, T, d_model)
        x = self.embedding(x)
        x = self.dropout(x)

        # Apply Hydra blocks with residual connections
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x)
            x = norm(x)
            x = x + residual  # Residual connection

        # Final normalization
        x = self.final_norm(x)

        # Pool over time: (B, T, d_model) -> (B, d_model)
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.pool(x).squeeze(-1)  # (B, d_model)

        # Classify
        output = self.classifier(x)

        return output


# ================================================================================
# Testing
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("True Classical Hydra - Testing")
    print("=" * 80)

    # Test parameters matching PhysioNet EEG
    batch_size = 4
    n_channels = 64
    n_timesteps = 160
    d_model = 128
    d_state = 16
    output_dim = 2

    print("\n[1] Testing HydraBlock...")
    block = HydraBlock(d_model=d_model, d_state=d_state, expand=2)
    u = torch.randn(batch_size, n_timesteps, d_model)
    output = block(u)
    print(f"  Input shape: {u.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == u.shape, "HydraBlock output shape mismatch!"

    print("\n[2] Testing TrueClassicalHydra (full model)...")
    model = TrueClassicalHydra(
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        d_model=d_model,
        d_state=d_state,
        n_layers=2,
        output_dim=output_dim,
        dropout=0.1
    )

    x_input = torch.randn(batch_size, n_channels, n_timesteps)
    output = model(x_input)
    print(f"  Input shape: {x_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[0]}")
    assert output.shape == (batch_size, output_dim), "Model output shape mismatch!"

    print("\n[3] Checking trainable parameters...")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")

    print("\n[4] Testing gradient flow...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x_batch = torch.randn(batch_size, n_channels, n_timesteps)
    y_batch = torch.randint(0, output_dim, (batch_size,))

    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient norm (embedding): {model.embedding.weight.grad.norm().item():.4f}")

    print("\n[5] Testing bidirectional processing...")
    # Create a test sequence with a clear pattern
    test_d_model = 8  # Use larger d_model for testing
    test_seq = torch.zeros(1, 10, test_d_model)
    test_seq[0, 0, :] = 1.0  # Mark start timestep
    test_seq[0, -1, :] = 1.0  # Mark end timestep

    block_test = HydraBlock(d_model=test_d_model, d_state=4, expand=2)
    output_test = block_test(test_seq)
    print(f"  Input shape: {test_seq.shape}")
    print(f"  Output shape: {output_test.shape}")
    print(f"  First timestep output norm: {output_test[0, 0, :].norm().item():.4f}")
    print(f"  Last timestep output norm: {output_test[0, -1, :].norm().item():.4f}")
    print(f"  Note: Both should be non-zero due to bidirectional processing")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
    print("\nKey Features Implemented:")
    print("  ✓ Selective State Space Model (SSM) with A, B, C, D matrices")
    print("  ✓ Time-step modulation (dt) with softplus activation")
    print("  ✓ TRUE bidirectional processing (forward + flipped backward)")
    print("  ✓ Quasiseparable-inspired structure via SSM")
    print("  ✓ Gated outputs with RMSNorm")
    print("  ✓ Residual connections")
    print("  ✓ 1D convolution for local context")
    print("=" * 80)

"""Adaptive Hybrid Attention–Mamba–Convolution language model.

Input shape:  [batch, sequence_length]
Logits shape: [batch, sequence_length, vocab_size]
Gate shape per layer: [batch, sequence_length, 3]

The three branches are causal self-attention, Mamba selective SSM, and a
left-padded depthwise Conv1D. Their outputs are fused with input-dependent
softmax weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba


class AdaptiveFusion(nn.Module):
    """Token-wise softmax fusion of attention, Mamba, and convolution."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        hidden_dim = max(dim // 4, 1)
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_out, mamba_out, conv_out):
        weights = F.softmax(self.gate(x), dim=-1)
        fused = (
            weights[..., 0:1] * attention_out
            + weights[..., 1:2] * mamba_out
            + weights[..., 2:3] * conv_out
        )
        return self.dropout(fused), weights


class HybridBlock(nn.Module):
    """Pre-normalized hybrid block with causal branches and residual paths."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        conv_kernel: int = 3,
        mamba_state_dim: int = 16,
        mamba_conv_width: int = 4,
        mamba_expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if conv_kernel < 1:
            raise ValueError("conv_kernel must be >= 1")

        self.norm1 = nn.RMSNorm(dim)
        self.attn_proj = nn.Linear(dim, dim)
        self.mamba_proj = nn.Linear(dim, dim)
        self.conv_proj = nn.Linear(dim, dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Actual selective state-space model from mamba-ssm.
        self.mamba = Mamba(
            d_model=dim,
            d_state=mamba_state_dim,
            d_conv=mamba_conv_width,
            expand=mamba_expand,
        )
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=conv_kernel,
            padding=0,
            groups=dim,
        )
        self.fusion = AdaptiveFusion(dim, dropout=dropout)
        self.residual_dropout = nn.Dropout(dropout)

        self.norm2 = nn.RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        seq_len = x_norm.size(1)

        # True entries mask future keys for each query position.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        a = self.attn_proj(x_norm)
        attention_out, _ = self.attention(
            a, a, a, attn_mask=causal_mask, need_weights=False
        )

        # Mamba is a causal selective state-space sequence mixer.
        mamba_out = self.mamba(self.mamba_proj(x_norm))

        # Left-only padding prevents access to future tokens.
        c = self.conv_proj(x_norm).transpose(1, 2)
        c = F.pad(c, (self.conv.kernel_size[0] - 1, 0))
        conv_out = self.conv(c).transpose(1, 2)

        fused, gate_weights = self.fusion(
            x_norm, attention_out, mamba_out, conv_out
        )
        x = residual + self.residual_dropout(fused)
        x = x + self.ffn(self.norm2(x))
        return x, gate_weights


class HybridModel(nn.Module):
    """Causal next-token language model with learned positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 128,
        conv_kernel: int = 3,
        mamba_state_dim: int = 16,
        mamba_conv_width: int = 4,
        mamba_expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.embedding_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                HybridBlock(
                    dim=dim,
                    num_heads=num_heads,
                    conv_kernel=conv_kernel,
                    mamba_state_dim=mamba_state_dim,
                    mamba_conv_width=mamba_conv_width,
                    mamba_expand=mamba_expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}"
            )
        seq_len = input_ids.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        all_gates = []
        for layer in self.layers:
            x, gates = layer(x)
            all_gates.append(gates)

        logits = self.lm_head(self.norm(x))
        return logits, all_gates

"""Controlled branch-ablation variants for the hybrid language model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba


class AdaptiveFusion(nn.Module):
    """Token-wise learned softmax fusion; names preserve prior checkpoint keys."""
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        hidden = max(dim // 4, 1)
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_out, mamba_out, conv_out):
        weights = F.softmax(self.gate(x), dim=-1)
        fused = (weights[..., 0:1] * attention_out
                 + weights[..., 1:2] * mamba_out
                 + weights[..., 2:3] * conv_out)
        return self.dropout(fused), weights


class HybridBlock(nn.Module):
    def __init__(self, dim, num_heads, conv_kernel, mamba_state_dim,
                 mamba_conv_width, mamba_expand, dropout,
                 branch_mode="all", fusion_mode="adaptive"):
        super().__init__()
        if branch_mode not in {"all", "attention", "mamba", "conv"}:
            raise ValueError(f"Unsupported branch_mode: {branch_mode}")
        if fusion_mode not in {"adaptive", "equal"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.branch_mode = branch_mode
        self.fusion_mode = fusion_mode
        self.norm1 = nn.RMSNorm(dim)
        self.residual_dropout = nn.Dropout(dropout)

        # Instantiate only the branch(es) used by this variant.
        if branch_mode in {"all", "attention"}:
            self.attn_proj = nn.Linear(dim, dim)
            self.attention = nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, batch_first=True)
        if branch_mode in {"all", "mamba"}:
            self.mamba_proj = nn.Linear(dim, dim)
            self.mamba = Mamba(
                d_model=dim, d_state=mamba_state_dim,
                d_conv=mamba_conv_width, expand=mamba_expand)
        if branch_mode in {"all", "conv"}:
            self.conv_proj = nn.Linear(dim, dim)
            self.conv = nn.Conv1d(
                dim, dim, kernel_size=conv_kernel,
                padding=0, groups=dim)
        if branch_mode == "all" and fusion_mode == "adaptive":
            self.fusion = AdaptiveFusion(dim, dropout=dropout)
        self.residual_dropout = nn.Dropout(dropout)

        self.norm2 = nn.RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout))

    def forward(self, x):
        residual = x
        z = self.norm1(x)
        seq_len = z.size(1)
        outs = []
        weights = z.new_zeros((*z.shape[:2], 3))

        if self.branch_mode in {"all", "attention"}:
            a = self.attn_proj(z)
            mask = torch.triu(torch.ones(
                seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            a, _ = self.attention(a, a, a, attn_mask=mask, need_weights=False)
            outs.append(("attention", a))
        if self.branch_mode in {"all", "mamba"}:
            m = self.mamba(self.mamba_proj(z))
            outs.append(("mamba", m))
        if self.branch_mode in {"all", "conv"}:
            c = self.conv_proj(z).transpose(1, 2)
            c = F.pad(c, (self.conv.kernel_size[0] - 1, 0))
            c = self.conv(c).transpose(1, 2)
            outs.append(("conv", c))

        if self.branch_mode == "all":
            if self.fusion_mode == "adaptive":
                fused, weights = self.fusion(z, outs[0][1], outs[1][1], outs[2][1])
            else:
                weights = z.new_full((*z.shape[:2], 3), 1.0 / 3.0)
                fused = (weights[..., 0:1] * outs[0][1]
                         + weights[..., 1:2] * outs[1][1]
                         + weights[..., 2:3] * outs[2][1])
        else:
            name, out = outs[0]
            fused = out
            weights[..., {"attention": 0, "mamba": 1, "conv": 2}[name]] = 1.0

        x = residual + self.residual_dropout(fused)
        x = x + self.ffn(self.norm2(x))
        return x, weights


class HybridModel(nn.Module):
    def __init__(self, vocab_size, dim=128, num_layers=4, num_heads=4,
                 max_seq_len=128, conv_kernel=3, mamba_state_dim=16,
                 mamba_conv_width=4, mamba_expand=2, dropout=0.0,
                 branch_mode="all", fusion_mode="adaptive"):
        super().__init__()
        self.branch_mode = branch_mode
        self.fusion_mode = fusion_mode
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            HybridBlock(dim, num_heads, conv_kernel, mamba_state_dim,
                        mamba_conv_width, mamba_expand, dropout,
                        branch_mode, fusion_mode)
            for _ in range(num_layers)])
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have [batch, sequence] shape")
        seq_len = input_ids.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} > {self.max_seq_len}")
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding_dropout(
            self.token_embedding(input_ids) + self.position_embedding(pos))
        gates = []
        for layer in self.layers:
            x, w = layer(x)
            gates.append(w)
        return self.lm_head(self.norm(x)), gates

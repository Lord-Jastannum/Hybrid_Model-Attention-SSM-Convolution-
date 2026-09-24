import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 3)
        )

    def forward(self, x, attention_out, ssm_out, conv_out):

        weights = self.gate(x)
        weights = F.softmax(weights, dim=-1)

        g_a = weights[..., 0:1]
        g_s = weights[..., 1:2]
        g_c = weights[..., 2:3]

        fused = (
            g_a * attention_out +
            g_s * ssm_out +
            g_c * conv_out
        )

        return fused, weights


class HybridBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        conv_kernel=3
    ):
        super().__init__()

        self.norm1 = nn.RMSNorm(dim)

        # Branch projections
        self.attn_proj = nn.Linear(dim, dim)
        self.ssm_proj = nn.Linear(dim, dim)
        self.conv_proj = nn.Linear(dim, dim)

        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # SSM placeholder
        # We will replace this with a real
        # Mamba implementation next.
        self.ssm = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

        # Local convolution
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=dim
        )

        # Adaptive fusion
        self.fusion = AdaptiveFusion(dim)

        # FFN
        self.norm2 = nn.RMSNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):

        residual = x

        x_norm = self.norm1(x)

        # -----------------------
        # Attention branch
        # -----------------------

        a = self.attn_proj(x_norm)

        attention_out, _ = self.attention(
            a, a, a
        )

        # -----------------------
        # SSM branch
        # -----------------------

        s = self.ssm_proj(x_norm)

        ssm_out = self.ssm(s)

        # -----------------------
        # Convolution branch
        # -----------------------

        c = self.conv_proj(x_norm)

        c = c.transpose(1, 2)

        conv_out = self.conv(c)

        conv_out = conv_out.transpose(1, 2)

        # -----------------------
        # Adaptive fusion
        # -----------------------

        fused, gate_weights = self.fusion(
            x_norm,
            attention_out,
            ssm_out,
            conv_out
        )

        # Residual
        x = residual + fused

        # -----------------------
        # FFN
        # -----------------------

        x = x + self.ffn(
            self.norm2(x)
        )

        return x, gate_weights

        class HybridModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        dim=512,
        num_layers=12,
        num_heads=8
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            dim
        )

        self.layers = nn.ModuleList([
            HybridBlock(
                dim=dim,
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.RMSNorm(dim)

        self.lm_head = nn.Linear(
            dim,
            vocab_size,
            bias=False
        )

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        all_gates = []

        for layer in self.layers:

            x, gates = layer(x)

            all_gates.append(gates)

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits, all_gates
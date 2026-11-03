
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
        # Compute input-dependent branch weights.
        weights = self.gate(x)
        weights = F.softmax(weights, dim=-1)

        g_a = weights[..., 0:1]
        g_s = weights[..., 1:2]
        g_c = weights[..., 2:3]

        # Adaptive three-way fusion.
        fused = (
            g_a * attention_out
            + g_s * ssm_out
            + g_c * conv_out
        )

        return fused, weights


class HybridBlock(nn.Module):
    def __init__(self, dim, num_heads=8, conv_kernel=3):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                "dim must be divisible by num_heads."
            )

        self.norm1 = nn.RMSNorm(dim)

        # Branch-specific projections.
        self.attn_proj = nn.Linear(dim, dim)
        self.ssm_proj = nn.Linear(dim, dim)
        self.conv_proj = nn.Linear(dim, dim)

        # Attention branch.
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Temporary SSM placeholder.
        # This is NOT real Mamba or a true selective SSM.
        self.ssm = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

        # Depthwise causal convolution.
        # Padding is applied manually in forward().
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=conv_kernel,
            padding=0,
            groups=dim
        )

        # Adaptive three-way fusion.
        self.fusion = AdaptiveFusion(dim)

        # Feed-forward network.
        self.norm2 = nn.RMSNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        residual = x

        x_norm = self.norm1(x)

        # --------------------------------------------------
        # 1. Causal attention branch
        # --------------------------------------------------
        a = self.attn_proj(x_norm)

        seq_len = a.size(1)

        # True entries block attention to future positions.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=a.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        attention_out, _ = self.attention(
            a,
            a,
            a,
            attn_mask=causal_mask,
            need_weights=False
        )

        # --------------------------------------------------
        # 2. Temporary SSM branch
        # --------------------------------------------------
        s = self.ssm_proj(x_norm)
        ssm_out = self.ssm(s)

        # --------------------------------------------------
        # 3. Causal local convolution branch
        # --------------------------------------------------
        c = self.conv_proj(x_norm)
        c = c.transpose(1, 2)

        # Left padding ensures position t cannot access
        # convolution inputs from positions greater than t.
        left_padding = self.conv.kernel_size[0] - 1
        c = F.pad(c, (left_padding, 0))

        conv_out = self.conv(c)
        conv_out = conv_out.transpose(1, 2)

        # --------------------------------------------------
        # 4. Adaptive three-way fusion
        # --------------------------------------------------
        fused, gate_weights = self.fusion(
            x_norm,
            attention_out,
            ssm_out,
            conv_out
        )

        # First residual connection.
        x = residual + fused

        # Feed-forward network and second residual.
        x = x + self.ffn(self.norm2(x))

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

        # Token embeddings.
        self.embedding = nn.Embedding(
            vocab_size,
            dim
        )

        # Stack of hybrid blocks.
        self.layers = nn.ModuleList([
            HybridBlock(
                dim=dim,
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])

        # Final normalization.
        self.norm = nn.RMSNorm(dim)

        # Language-model output head.
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

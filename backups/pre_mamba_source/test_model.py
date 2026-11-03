"""Smoke tests for the real-Mamba hybrid model. Run with: python test_model.py"""
import torch
from model import HybridModel
import config


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This Mamba installation is configured for CUDA. Activate mamba_env "
            "inside WSL and confirm torch.cuda.is_available() is True."
        )

    torch.manual_seed(config.SEED)
    device = config.DEVICE
    model = HybridModel(
        vocab_size=config.VOCAB_SIZE,
        dim=config.MODEL_DIM,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        max_seq_len=config.SEQUENCE_LENGTH,
        conv_kernel=config.CONV_KERNEL,
        mamba_state_dim=config.MAMBA_STATE_DIM,
        mamba_conv_width=config.MAMBA_CONV_WIDTH,
        mamba_expand=config.MAMBA_EXPAND,
        dropout=config.DROPOUT,
    ).to(device)

    model.train()
    input_ids = torch.randint(
        0, config.VOCAB_SIZE, (2, 32), device=device
    )
    logits, gates = model(input_ids)
    assert logits.shape == (2, 32, config.VOCAB_SIZE), logits.shape
    assert len(gates) == config.NUM_LAYERS
    assert gates[0].shape == (2, 32, 3), gates[0].shape
    assert torch.allclose(
        gates[0].sum(dim=-1), torch.ones_like(gates[0][..., 0]), atol=1e-5
    )

    loss = logits.float().square().mean()
    loss.backward()
    gradients = [
        p.grad for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert gradients, "No trainable parameter received a gradient"
    assert all(torch.isfinite(grad).all().item() for grad in gradients), (
        "At least one gradient contains NaN or infinity"
    )
    torch.cuda.synchronize()

    # Causality check: changing only future tokens must not change earlier logits.
    model.eval()
    prefix = torch.randint(0, config.VOCAB_SIZE, (1, 16), device=device)
    changed = prefix.clone()
    changed[:, 10:] = torch.randint(0, config.VOCAB_SIZE, (1, 6), device=device)
    with torch.no_grad():
        logits_a, _ = model(prefix)
        logits_b, _ = model(changed)
    if not torch.allclose(logits_a[:, :10], logits_b[:, :10], atol=2e-4, rtol=2e-4):
        max_diff = (logits_a[:, :10] - logits_b[:, :10]).abs().max().item()
        raise AssertionError(f"Causal leakage detected; max prefix difference={max_diff}")

    print("PASS: real-Mamba hybrid model smoke test")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Input shape:", tuple(input_ids.shape))
    print("Logits shape:", tuple(logits.shape))
    print("Gate shape per layer:", tuple(gates[0].shape))
    print("Gate sums, backward pass, and causal-prefix test passed.")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == "__main__":
    main()

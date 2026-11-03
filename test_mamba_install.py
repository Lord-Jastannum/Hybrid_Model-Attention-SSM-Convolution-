import torch
from mamba_ssm.modules.mamba_simple import Mamba

def main():
    assert torch.cuda.is_available(), "CUDA is not available"

    device = "cuda"
    model = Mamba(
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
    ).to(device)

    x = torch.randn(
        2, 128, 128,
        device=device,
        requires_grad=True,
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Running forward pass...")
    y = model(x)
    print("Output shape:", tuple(y.shape))

    loss = y.float().square().mean()
    print("Running backward pass...")
    loss.backward()
    torch.cuda.synchronize()

    assert y.shape == x.shape
    assert x.grad is not None

    print("PASS: Mamba forward and backward executed on CUDA.")

if __name__ == "__main__":
    main()
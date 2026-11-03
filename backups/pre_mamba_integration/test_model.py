import torch

from model import HybridModel


vocab_size = 10000

model = HybridModel(
    vocab_size=vocab_size,
    dim=512,
    num_layers=12,
    num_heads=8
)

input_ids = torch.randint(
    0,
    vocab_size,
    (2, 128)
)

logits, gates = model(input_ids)

print("Input shape:", input_ids.shape)
print("Logits shape:", logits.shape)
print("Gate shape:", gates[0].shape)

print(
    "Average gate weights:",
    gates[0].mean(dim=(0, 1))
)
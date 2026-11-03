# Architecture and implementation notes

## 1. Task and tensor conventions

The model performs autoregressive next-token prediction. Given input IDs with
shape `[batch, sequence_length]`, it returns logits with shape
`[batch, sequence_length, vocabulary_size]`. Training shifts each prepared block:
`input_ids = sequence[:, :-1]`, `targets = sequence[:, 1:]`.

## 2. Embedding stage

Token embeddings and learned absolute positional embeddings are added before the
hybrid stack. Inputs longer than `max_seq_len` are rejected explicitly.

## 3. Hybrid block

Each block applies RMSNorm before the three branches:

1. **Attention:** projected representations enter multi-head self-attention with a
   boolean upper-triangular mask. Future key positions are masked.
2. **Mamba:** projected representations enter `mamba_ssm.modules.mamba_simple.Mamba`,
   configured with `d_state=16`, `d_conv=4`, and `expand=2` by default.
3. **Convolution:** projected representations are transposed to channel-first form,
   padded only on the left by `kernel_size - 1`, and processed with depthwise Conv1D.

All branch outputs have shape `[batch, sequence_length, dim]`.

## 4. Adaptive fusion

A small MLP maps the normalized hidden state at each token to three logits. Softmax
produces weights `g_a`, `g_m`, and `g_c` that are nonnegative and sum to one:

`H = g_a * A + g_m * M + g_c * C`

The fused representation is added to the block input. A second RMSNorm and SiLU
feed-forward network are applied with another residual connection.

## 5. Output stage

Final RMSNorm and a linear language-model head produce next-token logits. The
implementation does not tie the token embedding and output-head weights.

## 6. Causality and tests

Attention uses a future-token mask. Convolution uses left-only padding. Mamba is a
causal sequence mixer. `test_model.py` checks output shapes, gate normalization,
backpropagation, and prefix invariance when only future input tokens are changed.
These tests must pass on the target CUDA environment before training.

## 7. Default experiment settings

- Vocabulary size: 10,000 (must match saved tokenizer metadata)
- Context length: 128
- Hidden dimension: 128
- Hybrid blocks: 4
- Attention heads: 4
- Mamba state dimension: 16
- Mamba convolution width: 4
- Mamba expansion: 2
- Batch size: 4
- Epochs: 5
- Optimizer: AdamW, learning rate `3e-4`, weight decay `0.01`
- Gradient clipping: `1.0`

These are initial research settings, not a claim of optimality. The 4 GB GPU may
require a lower batch size if memory usage exceeds available VRAM.

## 8. Outputs

All artifacts for this experiment are isolated under `outputs/mamba_hybrid_v1/`:

- `checkpoints/best_model.pt`
- `logs/training_history.csv`
- `logs/run_metadata.json`
- `logs/test_results.json`
- `plots/loss_curve.png`

## 9. Research limitations

The hybrid model should be evaluated against matched baselines (for example,
attention-only, Mamba-only, convolution-only, and hybrid ablations) using the same
tokenizer, data splits, context length, and comparable training budgets. Report
multiple seeds where feasible. Mean gate weights alone do not establish causal
importance or branch superiority. Document hardware, software versions, runtime,
peak GPU memory, and all hyperparameters in any final report.

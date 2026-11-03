# Adaptive Hybrid Attention–Mamba–Convolution Language Model

This experiment implements a causal next-token language model with three parallel
sequence-mixing branches: causal multi-head self-attention, the real Mamba selective
state-space module from `mamba_ssm`, and left-padded depthwise Conv1D. A token-wise
softmax gate fuses their outputs. The block uses RMSNorm, residual connections, and
an SiLU feed-forward network. Learned positional embeddings are added to token
embeddings.

## Important experiment isolation

This package writes new outputs only to `outputs/mamba_hybrid_v1/`. It does not
write to the original baseline's `outputs/checkpoints/best_model.pt`,
`outputs/logs/training_history.csv`, or `outputs/plots/loss_curve.png`.

## Files

- `model.py`: model architecture and adaptive fusion.
- `config.py`: data paths, model hyperparameters, and isolated output paths.
- `dataset.py`: existing WikiText-2 download/tokenization/chunking pipeline.
- `test_model.py`: GPU forward/backward, gate normalization, and causal-prefix tests.
- `train.py`: training, validation, best-checkpoint saving, final test evaluation,
  gate diagnostics, CSV history, JSON metadata/results, and loss plot.
- `requirements.txt`, `INSTALL.md`: Python dependencies and environment guidance.
- `ARCHITECTURE.md`: implementation and experimental design notes.

## Setup and run

1. Use the existing WSL environment and read `INSTALL.md`.
2. Confirm the current tokenizer and processed dataset exist under `data/`.
3. Run `python test_model.py` and require all checks to pass.
4. Run `python train.py` to start the Mamba experiment.

Do not rerun `dataset.py` unless you intend to rebuild the tokenizer and all
processed splits. The model vocabulary size must match the saved tokenizer.

## Reproducibility

The training seed, model hyperparameters, PyTorch/CUDA versions, device, and
parameter count are written to `outputs/mamba_hybrid_v1/logs/run_metadata.json`.
Epoch-level loss, validation perplexity, timing, and peak allocated GPU memory
are recorded in `training_history.csv`; final held-out test results are written to
`test_results.json`.

## Interpretation

A lower test perplexity indicates better next-token prediction only when
comparisons use the same dataset split, tokenizer, context length, and a fair
training/evaluation protocol. Gate averages are diagnostics, not proof that one
branch is intrinsically superior. This model's results must be compared with
properly trained baselines and ablations before making superiority claims.

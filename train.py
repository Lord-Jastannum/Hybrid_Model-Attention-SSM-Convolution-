"""Train and evaluate the real-Mamba adaptive hybrid language model."""
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tokenizers import Tokenizer

import config
from model import HybridModel


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run dataset.py first.")
    sequences = torch.load(path, map_location="cpu", weights_only=True)
    if sequences.ndim != 2:
        raise ValueError(f"Expected 2D tensor in {path}, got {tuple(sequences.shape)}")
    expected_width = config.SEQUENCE_LENGTH + 1
    if sequences.shape[1] != expected_width:
        raise ValueError(
            f"{path.name} width={sequences.shape[1]}, expected {expected_width}; "
            "check config.SEQUENCE_LENGTH and the processed dataset."
        )
    return TensorDataset(sequences)


def create_loader(path, batch_size, shuffle):
    return DataLoader(
        load_split(path), batch_size=batch_size, shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE.type == "cuda"), drop_last=False,
    )


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    gate_sum = torch.zeros(3, dtype=torch.float64)
    gate_positions = 0
    with torch.no_grad():
        for (sequences,) in loader:
            sequences = sequences.to(config.DEVICE, non_blocking=True)
            input_ids, targets = sequences[:, :-1], sequences[:, 1:]
            logits, all_gates = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            count = targets.numel()
            total_loss += loss.item() * count
            total_tokens += count
            if all_gates:
                # Average gates across batch, positions, and layers, weighted by positions.
                stacked = torch.stack(all_gates, dim=0).detach().double()
                gate_sum += stacked.sum(dim=(0, 1, 2)).cpu()
                gate_positions += stacked.shape[0] * stacked.shape[1] * stacked.shape[2]
    mean_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(mean_loss, 20.0))
    mean_gates = (gate_sum / max(gate_positions, 1)).tolist()
    return mean_loss, perplexity, mean_gates


def main():
    set_seed(config.SEED)
    if config.DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print("=" * 68)
    print("Adaptive Hybrid Attention–Mamba–Convolution | WikiText-2")
    print("Experiment:", config.EXPERIMENT_NAME)
    print("Device:", config.DEVICE)
    if config.DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not config.TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {config.TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_PATH))
    actual_vocab_size = tokenizer.get_vocab_size()
    print("Tokenizer vocabulary:", actual_vocab_size)
    if actual_vocab_size != config.VOCAB_SIZE:
        raise ValueError(
            f"Tokenizer vocab={actual_vocab_size}, config.VOCAB_SIZE={config.VOCAB_SIZE}. "
            "Update config.py to match the existing tokenizer; do not retrain the tokenizer."
        )

    train_loader = create_loader(config.TRAIN_PATH, config.BATCH_SIZE, True)
    validation_loader = create_loader(config.VALIDATION_PATH, config.EVAL_BATCH_SIZE, False)
    test_loader = create_loader(config.TEST_PATH, config.EVAL_BATCH_SIZE, False)

    model_kwargs = dict(
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
    )
    model = HybridModel(**model_kwargs).to(config.DEVICE)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Training batches/epoch: {len(train_loader):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    history = []
    best_validation_loss = float("inf")
    best_checkpoint = config.CHECKPOINT_DIR / "best_model.pt"
    history_path = config.LOG_DIR / "training_history.csv"
    metadata_path = config.LOG_DIR / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump({
            "experiment_name": config.EXPERIMENT_NAME,
            "model": model_kwargs,
            "trainable_parameters": parameter_count,
            "seed": config.SEED,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(config.DEVICE),
            "gpu_name": torch.cuda.get_device_name(0) if config.DEVICE.type == "cuda" else None,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS,
            "learning_rate": config.LEARNING_RATE,
        }, f, indent=2)

    fields = ["epoch", "train_loss", "validation_loss", "validation_perplexity", "epoch_seconds", "peak_gpu_memory_mb"]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config.EPOCHS + 1):
            model.train()
            if config.DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.time()
            total_loss = 0.0
            total_tokens = 0
            for step, (sequences,) in enumerate(train_loader, start=1):
                sequences = sequences.to(config.DEVICE, non_blocking=True)
                input_ids, targets = sequences[:, :-1], sequences[:, 1:]
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(input_ids)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={step}: {loss.item()}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()
                n = targets.numel()
                total_loss += loss.item() * n
                total_tokens += n
                if step == 1 or step % 100 == 0:
                    print(f"Epoch {epoch}/{config.EPOCHS} | step {step}/{len(train_loader)} | loss {loss.item():.4f}", flush=True)

            train_loss = total_loss / max(total_tokens, 1)
            val_loss, val_ppl, _ = evaluate(model, validation_loader, criterion)
            elapsed = time.time() - start
            peak_memory_mb = (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
                if config.DEVICE.type == "cuda" else 0.0
            )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_perplexity": val_ppl,
                "epoch_seconds": elapsed,
                "peak_gpu_memory_mb": peak_memory_mb,
            }
            history.append(row)
            writer.writerow(row)
            f.flush()
            print("\n" + "-" * 60)
            print(f"Epoch: {epoch}/{config.EPOCHS}")
            print(f"Train loss: {train_loss:.4f} | validation loss: {val_loss:.4f}")
            print(f"Validation perplexity: {val_ppl:.2f}")
            print(f"Epoch time: {elapsed:.1f}s | peak GPU allocated: {peak_memory_mb:.1f} MB")
            print("-" * 60, flush=True)

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": val_loss,
                    "model_config": model_kwargs,
                    "experiment_name": config.EXPERIMENT_NAME,
                    "trainable_parameters": parameter_count,
                }, best_checkpoint)
                print("Saved best checkpoint:", best_checkpoint)

    plt.figure(figsize=(8, 5))
    plt.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], marker="o", label="Train")
    plt.plot([r["epoch"] for r in history], [r["validation_loss"] for r in history], marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Real-Mamba Hybrid: WikiText-2 Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    loss_plot = config.PLOT_DIR / "loss_curve.png"
    plt.savefig(loss_plot, dpi=160)
    plt.close()

    checkpoint = torch.load(best_checkpoint, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_ppl, test_gates = evaluate(model, test_loader, criterion)
    result = {
        "experiment_name": config.EXPERIMENT_NAME,
        "best_epoch": checkpoint["epoch"],
        "test_loss": test_loss,
        "test_perplexity": test_ppl,
        "mean_gate_weights_attention_mamba_conv": test_gates,
        "trainable_parameters": parameter_count,
        "checkpoint": str(best_checkpoint),
        "history_csv": str(history_path),
        "loss_plot": str(loss_plot),
        "peak_gpu_memory_mb": max((r["peak_gpu_memory_mb"] for r in history), default=0),
    }
    results_path = config.LOG_DIR / "test_results.json"
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nFINAL TEST RESULTS")
    print(json.dumps(result, indent=2))
    print("Original baseline outputs were not modified.")


if __name__ == "__main__":
    main()

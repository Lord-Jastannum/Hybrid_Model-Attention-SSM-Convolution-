
"""Controlled architecture benchmark runner for the Adaptive Hybrid LM.

Features:
- Five architecture variants
- Multiple random seeds
- Preserves previous experiment results
- Skips completed variant/seed combinations by default
- Optional --overwrite to deliberately rerun experiments
- Saves combined CSV and JSON after every completed run
- Independent checkpoints and logs for every variant/seed
"""

import argparse
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import HybridModel


# ---------------------------------------------------------
# 1. BENCHMARK VARIANTS
# ---------------------------------------------------------

VARIANTS = {
    "attention_only": ("attention", "adaptive"),
    "mamba_only": ("mamba", "adaptive"),
    "conv_only": ("conv", "adaptive"),
    "equal_fusion": ("all", "equal"),
    "adaptive_hybrid": ("all", "adaptive"),
}


# ---------------------------------------------------------
# 2. REPRODUCIBILITY
# ---------------------------------------------------------

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


# ---------------------------------------------------------
# 3. DATASET LOADING
# ---------------------------------------------------------

def load_split(path):

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    if not isinstance(data, torch.Tensor):
        raise ValueError(f"Invalid dataset tensor: {path}")

    if data.ndim != 2 or data.size(1) < 2:
        raise ValueError(
            f"Expected [N, sequence_length + 1] tensor: {path}"
        )

    inputs = data[:, :-1].long()
    targets = data[:, 1:].long()

    return TensorDataset(inputs, targets)


# ---------------------------------------------------------
# 4. CSV AND JSON UTILITIES
# ---------------------------------------------------------

def write_csv(path, rows):

    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(rows)


def save_all_results(project, results_map):

    output_dir = project / "outputs" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = sorted(
        results_map.values(),
        key=lambda r: (
            r["variant"],
            int(r["seed"])
        )
    )

    json_path = output_dir / "benchmark_summary.json"
    csv_path = output_dir / "benchmark_summary.csv"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(results, f, indent=2)

    write_csv(csv_path, results)

    print(
        f"\nSaved {len(results)} total experiment results."
    )

    return results


def load_existing_results(project):

    output_dir = project / "outputs" / "benchmarks"

    results_map = {}

    if not output_dir.exists():
        return results_map

    # First recover results from individual experiment folders.
    for result_file in output_dir.glob(
        "*/seed_*/logs/test_results.json"
    ):

        try:

            with open(
                result_file,
                "r",
                encoding="utf-8"
            ) as f:

                result = json.load(f)

            key = (
                result["variant"],
                int(result["seed"])
            )

            results_map[key] = result

        except (json.JSONDecodeError, KeyError, OSError) as e:

            print(
                f"Warning: Could not recover {result_file}: {e}"
            )

    # Also recover historical results from the combined JSON.
    summary_file = output_dir / "benchmark_summary.json"

    if summary_file.exists():

        try:

            with open(
                summary_file,
                "r",
                encoding="utf-8"
            ) as f:

                old_results = json.load(f)

            for result in old_results:

                key = (
                    result["variant"],
                    int(result["seed"])
                )

                # Individual result files take precedence.
                if key not in results_map:
                    results_map[key] = result

        except (json.JSONDecodeError, KeyError, OSError) as e:

            print(
                f"Warning: Could not recover summary JSON: {e}"
            )

    return results_map


# ---------------------------------------------------------
# 5. MODEL EVALUATION
# ---------------------------------------------------------

def evaluate(model, loader, device):

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    gate_sum = torch.zeros(
        3,
        dtype=torch.float64
    )

    gate_count = 0

    with torch.inference_mode():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits, gates = model(x)

            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum"
            )

            total_loss += loss_sum.item()
            total_tokens += y.numel()

            if gates:

                stacked = torch.stack([
                    g.detach().mean((0, 1)).cpu()
                    for g in gates
                ]).mean(0)

                gate_sum += stacked.double()
                gate_count += 1

    mean_loss = total_loss / max(total_tokens, 1)

    perplexity = math.exp(
        min(mean_loss, 20)
    )

    if gate_count:

        mean_gates = (
            gate_sum / gate_count
        ).tolist()

    else:

        mean_gates = [
            None,
            None,
            None
        ]

    return (
        mean_loss,
        perplexity,
        mean_gates,
        total_tokens
    )


# ---------------------------------------------------------
# 6. INFERENCE PERFORMANCE BENCHMARK
# ---------------------------------------------------------

def benchmark_inference(
    model,
    loader,
    device,
    warmup=10,
    steps=40
):

    model.eval()

    batches = []

    for i, (x, _) in enumerate(loader):

        batches.append(x)

        if len(batches) >= warmup + steps:
            break

    if not batches:
        return {}

    warm_batches = batches[:min(warmup, len(batches))]

    timed_batches = (
        batches[len(warm_batches):]
        or batches[-1:]
    )

    if device.type == "cuda":

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    durations = []
    token_count = 0

    with torch.inference_mode():

        # Warm-up
        for x in warm_batches:

            model(x.to(device))

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed inference
        for x in timed_batches:

            x = x.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            duration = time.perf_counter() - start

            durations.append(duration)
            token_count += x.numel()

    elapsed = sum(durations)

    result = {

        "inference_tokens_per_sec":
            token_count / max(elapsed, 1e-12),

        "batch_latency_ms_median":
            statistics.median(durations) * 1000,

        "batch_latency_ms_p95":
            float(np.percentile(durations, 95)) * 1000,

        "inference_peak_allocated_mb":
            (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else None
            ),

        "inference_peak_reserved_mb":
            (
                torch.cuda.max_memory_reserved(device) / 1024**2
                if device.type == "cuda"
                else None
            )
    }

    return result


# ---------------------------------------------------------
# 7. TRAIN ONE VARIANT FOR ONE SEED
# ---------------------------------------------------------

def run_variant(
    args,
    name,
    seed,
    train_ds,
    val_ds,
    test_ds,
    device,
    project
):

    branch_mode, fusion_mode = VARIANTS[name]

    seed_everything(seed)

    output_dir = (
        project
        / "outputs"
        / "benchmarks"
        / name
        / f"seed_{seed}"
    )

    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 65)
    print(f"VARIANT: {name}")
    print(f"SEED: {seed}")
    print(f"BRANCH MODE: {branch_mode}")
    print(f"FUSION MODE: {fusion_mode}")
    print("=" * 65)

    # -----------------------------------------------------
    # Data loaders
    # -----------------------------------------------------

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = HybridModel(

        vocab_size=args.vocab_size,

        dim=args.dim,

        num_layers=args.layers,

        num_heads=args.heads,

        max_seq_len=args.seq_len,

        conv_kernel=args.conv_kernel,

        mamba_state_dim=args.mamba_state_dim,

        mamba_conv_width=args.mamba_conv_width,

        mamba_expand=args.mamba_expand,

        dropout=args.dropout,

        branch_mode=branch_mode,

        fusion_mode=fusion_mode

    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {parameters:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    history = []

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):

        model.train()

        total_train_loss = 0.0
        total_train_tokens = 0

        if device.type == "cuda":

            torch.cuda.reset_peak_memory_stats(device)

        start_time = time.perf_counter()

        for x, y in train_loader:

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits, _ = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, args.vocab_size),
                y.reshape(-1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip
            )

            optimizer.step()

            total_train_loss += (
                loss.item() * y.numel()
            )

            total_train_tokens += y.numel()

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_seconds = (
            time.perf_counter() - start_time
        )

        train_loss = (
            total_train_loss /
            max(total_train_tokens, 1)
        )

        val_loss, val_ppl, _, _ = evaluate(
            model,
            val_loader,
            device
        )

        allocated_mb = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        )

        reserved_mb = (
            torch.cuda.max_memory_reserved(device) / 1024**2
            if device.type == "cuda"
            else None
        )

        row = {

            "epoch": epoch,

            "train_loss": train_loss,

            "validation_loss": val_loss,

            "validation_perplexity": val_ppl,

            "epoch_seconds": epoch_seconds,

            "train_tokens_per_sec":
                total_train_tokens / max(epoch_seconds, 1e-12),

            "peak_allocated_mb": allocated_mb,

            "peak_reserved_mb": reserved_mb
        }

        history.append(row)

        print(
            f"{name} | "
            f"Seed {seed} | "
            f"Epoch {epoch}/{args.epochs} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"PPL {val_ppl:.2f} | "
            f"Tokens/s {row['train_tokens_per_sec']:.1f} | "
            f"Time {epoch_seconds:.1f}s",
            flush=True
        )

        # Save best model based on validation loss.
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(

                {
                    "model_state_dict": model.state_dict(),

                    "variant": name,

                    "seed": seed,

                    "epoch": epoch,

                    "val_loss": val_loss,

                    "config": vars(args)
                },

                checkpoint_dir / "best_model.pt"
            )

    # Save training history.
    write_csv(
        log_dir / "training_history.csv",
        history
    )

    # -----------------------------------------------------
    # Load best checkpoint
    # -----------------------------------------------------

    checkpoint_path = (
        checkpoint_dir / "best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # -----------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------

    test_loss, test_ppl, gate_weights, test_tokens = evaluate(
        model,
        test_loader,
        device
    )

    # -----------------------------------------------------
    # Inference benchmark
    # -----------------------------------------------------

    inference_metrics = benchmark_inference(
        model,
        test_loader,
        device
    )

    peak_train_allocated = max(
        (
            row["peak_allocated_mb"] or 0
            for row in history
        ),
        default=0
    )

    peak_train_reserved = max(
        (
            row["peak_reserved_mb"] or 0
            for row in history
        ),
        default=0
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    result = {

        "variant": name,

        "branch_mode": branch_mode,

        "fusion_mode": fusion_mode,

        "seed": seed,

        "parameters": parameters,

        "best_epoch": best_epoch,

        "best_validation_loss": best_val_loss,

        "test_loss": test_loss,

        "test_perplexity": test_ppl,

        "test_tokens": test_tokens,

        "mean_gate_attention": gate_weights[0],

        "mean_gate_mamba": gate_weights[1],

        "mean_gate_conv": gate_weights[2],

        "mean_epoch_seconds":
            statistics.mean(
                row["epoch_seconds"]
                for row in history
            ),

        "total_training_seconds":
            sum(
                row["epoch_seconds"]
                for row in history
            ),

        "peak_train_allocated_mb":
            peak_train_allocated,

        "peak_train_reserved_mb":
            peak_train_reserved,

        "checkpoint_mb":
            checkpoint_path.stat().st_size / 1024**2,

        **inference_metrics
    }

    result_path = (
        log_dir / "test_results.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(result, f, indent=2)

    print("\nRESULT:")
    print(json.dumps(result, indent=2))

    return result


# ---------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Controlled Adaptive Hybrid LM Benchmark"
    )

    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent
    )

    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS.keys()),
        default=list(VARIANTS.keys())
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=128
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=128
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--heads",
        type=int,
        default=4
    )

    parser.add_argument(
        "--conv-kernel",
        type=int,
        default=3
    )

    parser.add_argument(
        "--mamba-state-dim",
        type=int,
        default=16
    )

    parser.add_argument(
        "--mamba-conv-width",
        type=int,
        default=4
    )

    parser.add_argument(
        "--mamba-expand",
        type=int,
        default=2
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun and replace existing variant/seed results."
    )

    args = parser.parse_args()

    project = args.project_dir.resolve()

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("ADAPTIVE HYBRID LANGUAGE MODEL BENCHMARK")
    print("=" * 65)

    print(f"Device: {device}")

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(f"Variants: {args.variants}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Overwrite: {args.overwrite}")

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    data_dir = (
        project / "data" / "processed"
    )

    train_ds = load_split(
        data_dir / "train.pt"
    )

    val_ds = load_split(
        data_dir / "validation.pt"
    )

    test_ds = load_split(
        data_dir / "test.pt"
    )

    actual_seq_len = (
        train_ds.tensors[0].shape[1]
    )

    if actual_seq_len != args.seq_len:

        raise ValueError(
            f"Dataset sequence length is {actual_seq_len}, "
            f"but --seq-len is {args.seq_len}."
        )

    print("\nDATASET")
    print(f"Training sequences: {len(train_ds):,}")
    print(f"Validation sequences: {len(val_ds):,}")
    print(f"Test sequences: {len(test_ds):,}")

    # -----------------------------------------------------
    # Recover existing results
    # -----------------------------------------------------

    results_map = load_existing_results(project)

    print("\nEXISTING RESULTS")

    print(
        f"Recovered {len(results_map)} completed experiments."
    )

    # -----------------------------------------------------
    # Run experiments
    # -----------------------------------------------------

    for seed in args.seeds:

        for variant in args.variants:

            key = (
                variant,
                seed
            )

            if key in results_map and not args.overwrite:

                print(
                    f"\nSKIPPING: {variant}, seed={seed}"
                )

                print(
                    "A completed result already exists. "
                    "Use --overwrite to rerun it."
                )

                continue

            print(
                f"\nStarting experiment: "
                f"{variant}, seed={seed}"
            )

            result = run_variant(

                args=args,

                name=variant,

                seed=seed,

                train_ds=train_ds,

                val_ds=val_ds,

                test_ds=test_ds,

                device=device,

                project=project
            )

            # Update results immediately after each run.
            results_map[key] = result

            save_all_results(
                project,
                results_map
            )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    final_results = save_all_results(
        project,
        results_map
    )

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETED")
    print("=" * 65)

    print(
        f"Total preserved experiment results: "
        f"{len(final_results)}"
    )

    print(
        f"Results directory: "
        f"{project / 'outputs' / 'benchmarks'}"
    )

    print("\nAll completed results have been preserved.")


if __name__ == "__main__":

    main()

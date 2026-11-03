
"""
dataset.py

Downloads WikiText-2, trains a BPE tokenizer on the training split only,
and saves tokenized, fixed-length sequences for language-model training.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TOKENIZER_DIR = DATA_DIR / "tokenizer"

DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"


def get_nonempty_lines(split):
    """Return non-empty text lines from a dataset split."""
    return [
        line.strip()
        for line in split["text"]
        if line and line.strip()
    ]


def train_tokenizer(train_lines, vocab_size):
    """Train a byte-level BPE tokenizer using training text only."""
    tokenizer = Tokenizer(
        models.BPE(unk_token="<unk>")
    )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<pad>", "<unk>", "<eos>"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        iter(train_lines),
        trainer=trainer,
        length=len(train_lines),
    )

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_DIR / "tokenizer.json"))

    return tokenizer


def encode_split(tokenizer, lines, sequence_length):
    """
    Encode each non-empty line, append EOS, and create fixed-length
    blocks of sequence_length + 1 tokens.

    The extra token allows the training loop to shift inputs and targets.
    """
    eos_id = tokenizer.token_to_id("<eos>")

    if eos_id is None:
        raise ValueError("Tokenizer does not contain the <eos> token.")

    token_ids = []

    for line in lines:
        ids = tokenizer.encode(line).ids
        if ids:
            token_ids.extend(ids)
            token_ids.append(eos_id)

    block_size = sequence_length + 1
    usable_tokens = (len(token_ids) // block_size) * block_size

    token_ids = token_ids[:usable_tokens]

    if usable_tokens == 0:
        raise ValueError(
            "No complete sequences were generated. "
            "Check the dataset and sequence length."
        )

    chunks = torch.tensor(
        token_ids,
        dtype=torch.long,
    ).view(-1, block_size)

    return chunks, len(token_ids)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    if args.vocab_size < 300:
        raise ValueError("Use a vocabulary size of at least 300.")

    if args.sequence_length < 2:
        raise ValueError("Sequence length must be at least 2.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading/loading WikiText-2...")

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        cache_dir=str(RAW_DIR),
    )

    train_lines = get_nonempty_lines(dataset["train"])
    validation_lines = get_nonempty_lines(dataset["validation"])
    test_lines = get_nonempty_lines(dataset["test"])

    print(f"Training text lines:   {len(train_lines):,}")
    print(f"Validation text lines: {len(validation_lines):,}")
    print(f"Test text lines:       {len(test_lines):,}")

    print("\nTraining tokenizer on training split only...")

    tokenizer = train_tokenizer(
        train_lines,
        args.vocab_size,
    )

    actual_vocab_size = tokenizer.get_vocab_size()

    print(f"Tokenizer vocabulary size: {actual_vocab_size:,}")

    # Keep the model's embedding/output dimensions compatible with
    # the actual tokenizer vocabulary.
    metadata = {
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "source": (
            "https://huggingface.co/datasets/Salesforce/wikitext"
        ),
        "tokenizer": "Byte-level BPE",
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "sequence_length": args.sequence_length,
        "special_tokens": {
            "pad": tokenizer.token_to_id("<pad>"),
            "unk": tokenizer.token_to_id("<unk>"),
            "eos": tokenizer.token_to_id("<eos>"),
        },
        "splits": {},
    }

    splits = {
        "train": train_lines,
        "validation": validation_lines,
        "test": test_lines,
    }

    for split_name, lines in splits.items():
        print(f"\nEncoding {split_name} split...")

        chunks, token_count = encode_split(
            tokenizer,
            lines,
            args.sequence_length,
        )

        output_path = PROCESSED_DIR / f"{split_name}.pt"
        torch.save(chunks, output_path)

        metadata["splits"][split_name] = {
            "source_lines": len(lines),
            "tokens_after_chunking": token_count,
            "sequences": int(chunks.shape[0]),
            "output_file": str(output_path.relative_to(PROJECT_DIR)),
        }

        print(f"  Token count: {token_count:,}")
        print(f"  Sequences:   {chunks.shape[0]:,}")
        print(f"  Saved to:    {output_path}")

    metadata_path = PROCESSED_DIR / "metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nDataset preparation complete.")
    print(f"Tokenizer: {TOKENIZER_DIR / 'tokenizer.json'}")
    print(f"Metadata:  {metadata_path}")
    print(f"Actual vocabulary size: {actual_vocab_size:,}")
    print(
        "\nIMPORTANT: Configure the model's vocab_size to match "
        "actual_vocab_size before training if they differ."
    )


if __name__ == "__main__":
    main()

"""Configuration for the separate real-Mamba experiment."""
from pathlib import Path
import torch

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
TOKENIZER_PATH = PROJECT_DIR / "data" / "tokenizer" / "tokenizer.json"

# Keep the Mamba experiment separate from the original baseline outputs.
EXPERIMENT_NAME = "mamba_hybrid_v1"
OUTPUT_DIR = PROJECT_DIR / "outputs" / EXPERIMENT_NAME
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
PLOT_DIR = OUTPUT_DIR / "plots"

TRAIN_PATH = DATA_DIR / "train.pt"
VALIDATION_PATH = DATA_DIR / "validation.pt"
TEST_PATH = DATA_DIR / "test.pt"

VOCAB_SIZE = 10_000  # Must equal tokenizer.get_vocab_size().
SEQUENCE_LENGTH = 128
MODEL_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 4
CONV_KERNEL = 3
MAMBA_STATE_DIM = 16
MAMBA_CONV_WIDTH = 4
MAMBA_EXPAND = 2
DROPOUT = 0.0

BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for directory in (CHECKPOINT_DIR, LOG_DIR, PLOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


from pathlib import Path
import torch

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
TOKENIZER_PATH = PROJECT_DIR / "data" / "tokenizer" / "tokenizer.json"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
PLOT_DIR = OUTPUT_DIR / "plots"

# Dataset
VOCAB_SIZE = 10_000
SEQUENCE_LENGTH = 128

TRAIN_PATH = DATA_DIR / "train.pt"
VALIDATION_PATH = DATA_DIR / "validation.pt"
TEST_PATH = DATA_DIR / "test.pt"

# Small initial experiment.
# We can scale this up after checking memory and training speed.
MODEL_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 4

# Training
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Evaluation
EVAL_BATCH_SIZE = 4
EVAL_EVERY_EPOCH = True

# Reproducibility
SEED = 42

# Prefer CUDA when available.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Create output directories
for directory in (
    CHECKPOINT_DIR,
    LOG_DIR,
    PLOT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

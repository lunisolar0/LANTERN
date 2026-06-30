"""
LANTERN configuration and hyperparameters.
"""
import os

# ── Paths ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

# ── Dataset ──
DATASETS = ["gc", "mc", "mf"]  # Google Core, Mozilla Core, Mozilla Firefox

# ── Text embedding ──
EMBED_DIM = 128          # TF-IDF + SVD reduced dimension
TFIDF_MAX_FEATURES = 5000

# ── Model ──
HIDDEN_DIM = 64          # hidden / embedding dimension for developer nodes
NUM_LAYERS_EXP = 3       # explicit bipartite propagation layers (Le)
NUM_LAYERS_IMP = 2       # implicit developer-projected layers (Li)
TOP_K_AUG = 5            # Top-K developers for semantic augmentation
TAU_MIN = 0.30           # minimum cosine similarity threshold for augmentation

# ── Training ──
EPOCHS = 200
BATCH_SIZE = 2048
LR = 1e-3
L2_REG = 1e-4            # weight decay on structural embeddings only
NEG_SAMPLES = 1          # negative samples per positive in BPR
EARLY_STOP_PATIENCE = 20

# ── Evaluation ──
TOP_K_EVAL = [1, 3, 5, 10]

# ── Device ──
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
LANTERN: Latent Relationship Mining and Atomicity-Aware Dual-View Routing
for Load-Balanced Bug Triage.

Reference:
    "LANTERN: Latent Relationship Mining and Atomicity-Aware Dual-View Routing
     for Load-Balanced Bug Triage"
"""

from .config import (
    DATA_DIR, DATASETS, EMBED_DIM, TFIDF_MAX_FEATURES,
    HIDDEN_DIM, NUM_LAYERS_EXP, NUM_LAYERS_IMP,
    TOP_K_AUG, TAU_MIN, EPOCHS, LR, L2_REG,
    EARLY_STOP_PATIENCE, DEVICE, TOP_K_EVAL,
)
from .dataset import BipartiteDataset, set_seed
from .model import LANTERN, build_augmented_adjacency
from .utils import compute_metrics, evaluate_model, format_metrics

__version__ = "1.0.0"

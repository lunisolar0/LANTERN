"""
Data loading and preprocessing for LANTERN.

Builds the bug-developer bipartite graph from raw issue tracker records,
computes frozen text embeddings via TF-IDF + truncated SVD, and provides
train / validation / test splits.
"""
import json
import os
import random
import pickle
from collections import defaultdict

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from .config import DATA_DIR, EMBED_DIM, TFIDF_MAX_FEATURES, DEVICE


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class BipartiteDataset:
    """
    Bug-developer bipartite graph built from issue tracker records.

    Each record is treated as a single bug report assigned to one developer.
    The raw data is expected in ``data/{name}.json`` with each entry containing
    ``owner``, ``issue_title``, and ``description`` fields.

    Preprocessed artefacts (index mappings, embeddings, splits) are cached to
    ``data/{name}_cache/processed.pkl`` for fast subsequent loads.
    """

    def __init__(self, dataset_name, cache_dir=None):
        self.name = dataset_name
        self.cache_dir = cache_dir or os.path.join(DATA_DIR, f"{dataset_name}_cache")

        raw_path = os.path.join(DATA_DIR, f"{dataset_name}.json")
        cache_path = os.path.join(self.cache_dir, "processed.pkl")

        if os.path.exists(cache_path):
            print(f"[{dataset_name}] Loading cached data from {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.__dict__.update(cached)
        else:
            print(f"[{dataset_name}] Loading raw data from {raw_path}")
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            self._process(raw_data)
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self.__dict__, f)

    # ------------------------------------------------------------------
    #  Data processing pipeline
    # ------------------------------------------------------------------

    def _process(self, raw_data):
        """Run the full preprocessing pipeline: index → split → graph → embeddings."""
        owner_to_idx = {}
        bug_texts = []
        bug_owners = []

        for entry in tqdm(raw_data, desc=f"[{self.name}] Indexing"):
            owner = entry["owner"]
            if owner not in owner_to_idx:
                owner_to_idx[owner] = len(owner_to_idx)
            bug_owners.append(owner_to_idx[owner])
            text = entry["issue_title"] + " " + entry["description"]
            bug_texts.append(text)

        self.num_bugs = len(bug_texts)
        self.num_devs = len(owner_to_idx)
        self.owner_to_idx = owner_to_idx
        self.idx_to_owner = {v: k for k, v in owner_to_idx.items()}
        self.bug_owners = np.array(bug_owners, dtype=np.int64)

        print(f"[{self.name}] {self.num_bugs} bugs, {self.num_devs} developers")

        # ── Random 80 / 10 / 10 split ──
        indices = np.random.permutation(self.num_bugs)
        train_end = int(0.8 * self.num_bugs)
        val_end = int(0.9 * self.num_bugs)
        self.train_idx = indices[:train_end]
        self.val_idx = indices[train_end:val_end]
        self.test_idx = indices[val_end:]

        print(
            f"[{self.name}] Split: "
            f"train={len(self.train_idx)}, "
            f"val={len(self.val_idx)}, "
            f"test={len(self.test_idx)}"
        )

        self._build_adjacency()
        self._compute_embeddings(bug_texts)

    # ------------------------------------------------------------------
    #  Sparse adjacency construction
    # ------------------------------------------------------------------

    def _build_adjacency(self):
        """Build the sparse bug-developer adjacency matrix A from training edges."""
        dev_bugs = defaultdict(list)
        for bug_idx in self.train_idx:
            dev = self.bug_owners[bug_idx]
            dev_bugs[dev].append(bug_idx)

        rows, cols = [], []
        for bug_idx in self.train_idx:
            dev = self.bug_owners[bug_idx]
            rows.append(int(bug_idx))
            cols.append(int(dev))

        self.adj_rows = np.array(rows, dtype=np.int64)
        self.adj_cols = np.array(cols, dtype=np.int64)
        self.adj_data = np.ones(len(rows), dtype=np.float32)
        self.dev_to_bugs = dev_bugs

        dev_degree = np.zeros(self.num_devs, dtype=np.int64)
        for dev, bugs in dev_bugs.items():
            dev_degree[dev] = len(bugs)
        self.dev_degree = dev_degree

    # ------------------------------------------------------------------
    #  Frozen text embeddings  (X_b in the paper)
    # ------------------------------------------------------------------

    def _compute_embeddings(self, bug_texts):
        """
        Compute frozen semantic embeddings X_b via:

          1. TF-IDF vectorisation (sublinear, English stopwords)
          2. Truncated SVD reduction to ``EMBED_DIM``
          3. L₂ normalisation (unit vectors for cosine similarity)
        """
        print(f"[{self.name}] Computing TF-IDF + SVD embeddings (dim={EMBED_DIM}) ...")

        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            stop_words="english",
            sublinear_tf=True,
        )
        tfidf = vectorizer.fit_transform(bug_texts)
        print(f"[{self.name}]   TF-IDF shape: {tfidf.shape}")

        svd = TruncatedSVD(n_components=EMBED_DIM, random_state=42)
        embeddings = svd.fit_transform(tfidf)
        print(f"[{self.name}]   Embedding shape: {embeddings.shape}")

        # L₂ normalise to unit vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embeddings = embeddings / norms

        self.bug_embeddings = torch.tensor(
            embeddings, dtype=torch.float32, device=DEVICE
        )
        self.vectorizer = vectorizer
        self.svd = svd

    # ------------------------------------------------------------------
    #  Public helpers
    # ------------------------------------------------------------------

    def get_train_adj(self):
        """Return the sparse training adjacency matrix A  [N_b × N_d]."""
        indices = torch.stack([
            torch.tensor(self.adj_rows, dtype=torch.long),
            torch.tensor(self.adj_cols, dtype=torch.long),
        ])
        values = torch.tensor(self.adj_data, dtype=torch.float32)
        adj = torch.sparse_coo_tensor(
            indices, values,
            size=(self.num_bugs, self.num_devs),
            device=DEVICE,
        ).coalesce()
        return adj

    def get_train_edges(self):
        """Return (bug_idx, dev_idx) pairs for all training edges."""
        return (
            torch.tensor(self.adj_rows, dtype=torch.long),
            torch.tensor(self.adj_cols, dtype=torch.long),
        )

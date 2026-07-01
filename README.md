# LANTERN

**La**tent Relatio**n**ship Mining and A**t**omicity-Awar**e** Dual-View **R**outing for Load-Balanced Bug Triage.

> Reference implementation accompanying the paper *"Breaking the Vicious Cycle: A Load-Intervention Network for Bug Triage(LANTERN)"*.

---

## Overview

LANTERN is an end-to-end load-intervention framework designed for real-world issue tracker scenarios. It addresses two fundamental operational bottlenecks in bug triage:

1. **Topology Fragmentation** caused by strict single-assignee atomicity, which leaves long-tail developers structurally isolated from message-passing pathways.
2. **The Matthew Effect** (Expertise Concentration Trap), where historically overloaded veterans dominate task distribution, starving less-active developers of opportunities.

LANTERN resolves these challenges through two complementary modules:

- **Latent Relationship Mining** — parameter-free semantic voting that injects confidence-weighted soft edges into the bipartite assignment graph, bypassing atomicity without inflating computational overhead.
- **Atomicity-Aware Dual-View Routing** — a dual-path propagation architecture with an adaptive, degree-aware fusion gate that dynamically balances workloads between veterans and long-tail developers.

The framework is exceptionally lightweight: with bug semantics fixed offline, the learnable component requires only `O(N_d * d + 2)` parameters, enabling seamless CI/CD deployment alongside advanced language models.

---

## Model Architecture

LANTERN builds upon a LightGCN backbone with three key innovations:

### 1. Latent Relationship Mining (Graph Augmentation)

**Semantic Centroid Profiling.** Each developer's technical capability domain is represented as the semantic centroid of their historically resolved bugs. To strictly prevent target leakage, the target query bug is explicitly excluded from the centroid computation (target-aware profiling).

**Semantic Voting & Soft Edge Injection.** For each incoming bug, cosine similarity is computed against all developer centroids. A Top-*K* selection with a minimum similarity threshold `τ_min` identifies candidate developers, and confidence-weighted soft edges are injected into the original adjacency matrix:

> **Â = A + (S ⊙ M)**

where **A** preserves factual assignments at unit weight and semantic pseudo-edges receive continuous weights in `(0, 1]`.

### 2. Atomicity-Aware Dual-View Routing

**Explicit Bipartite Path.** Standard LightGCN-style symmetric normalization and propagation over the augmented bipartite graph, preserving factual assignment evidence.

**Developer-Projected Implicit Path.** A homogeneous developer-side projection **R̂_dev = Âᵀ D̂_b⁻¹ Â** is extracted to structurally enfranchise isolated developers. Propagation through this projection activates latent collaborative capabilities without relying on high-degree veterans.

### 3. Dynamic Load-Balancing Gate

An adaptive gate `γⱼ = σ(w·log(degⱼ + 1) + b)` fuses the explicit and implicit developer representations. The gate relies exclusively on the *original* factual adjacency to determine workload, ensuring high-degree developers predominantly use explicit factual edges while structurally isolated developers are empowered by the implicit network.

### 4. Optimization

The model is trained with the Bayesian Personalized Ranking (BPR) objective. An L₂ penalty is applied exclusively to the structural developer embeddings, while the low-parameter gate scalars are excluded from weight decay to preserve full adaptive capacity.

---

## Quick Start

```bash
# Clone and enter the repository
cd LANTERN

# Install dependencies
pip install -r requirements.txt

# Run on all three datasets
python run.py

# Or run on a single dataset
python run.py --dataset mc
```

The script automatically downloads / loads data, preprocesses text into semantic embeddings, builds the augmented graph, trains the model, and reports standard ranking metrics.

---

## Installation

**Requirements:** Python 3.8+ and PyTorch 1.12+.

```bash
pip install -r requirements.txt
```

Core dependencies:

| Package       | Purpose                         |
|---------------|---------------------------------|
| `torch`       | Model, autograd, sparse ops     |
| `numpy`       | Numerical utilities             |
| `scikit-learn`| TF-IDF + SVD text embeddings    |
| `tqdm`        | Progress bars                   |

A GPU is recommended for the graph augmentation step but not strictly required — all operations fall back gracefully to CPU.

---

## Usage

### Training & Evaluation

```bash
python train.py gc    # Google Core
python train.py mc    # Mozilla Core
python train.py mf    # Mozilla Firefox
```

Each run performs:
1. Data loading and 80/10/10 train/val/test split
2. TF-IDF + TruncatedSVD text embedding (frozen)
3. Semantic-voting-based graph augmentation
4. Full-batch BPR training with early stopping
5. Evaluation (HR@K, NDCG@K, MRR) on the test set


## Repository Structure

```
LANTERN/
├── lantern/                 # Core package
│   ├── __init__.py          # Public API exports
│   ├── config.py            # Hyperparameters & paths
│   ├── dataset.py           # Data loading, graph construction, embeddings
│   ├── model.py             # LANTERN model + graph augmentation
│   └── utils.py             # Metrics (HR@K, NDCG@K, MRR)
├── data/                    # Datasets (JSON) and cached preprocessed files
├── train.py                 # Single-dataset training entry point
├── run.py                   # Multi-dataset experiment runner
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

### Key Classes

| Module | Class / Function | Role |
|--------|-----------------|------|
| `lantern.dataset` | `BipartiteDataset` | Loads raw JSON, builds sparse bipartite graph, computes frozen text embeddings |
| `lantern.model` | `LANTERN` | Full model: explicit + implicit propagation + adaptive fusion |
| `lantern.model` | `build_augmented_adjacency` | Pre-computes the augmented adjacency via semantic voting |
| `lantern.utils` | `evaluate_model` | Computes HR@K, NDCG@K, MRR on val/test splits |

---

## Datasets

The repository supports three large-scale issue tracker datasets:

| Dataset | Code | Source | Approx. Records |
|---------|------|--------|----------------|
| Google Core | `gc` | Chromium issue tracker | ~460K |
| Mozilla Core | `mc` | Bugzilla Core product | ~499K |
| Mozilla Firefox | `mf` | Bugzilla Firefox product | ~89K |

Each record contains an `owner` (developer identifier), `issue_title`, and `description`. The processed JSON files are expected under `data/{gc,mc,mf}.json`.

---

## License

This project is released for research and reproducibility purposes. See the repository's license file for details.

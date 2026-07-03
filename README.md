# LANTERN

**L**atent **A**ugmentation **N**etwork for **T**riage with **E**xplicit-Implicit **R**outi**N**g.

> Reference implementation accompanying the paper *"Breaking the Vicious Cycle: A Load-Intervention Network for Bug Triage"*.

---

## 💡 Overview

LANTERN is a lightweight, end-to-end load-intervention framework designed for real-world issue tracker scenarios. It mitigates **Topology Fragmentation** and the **Matthew Effect** (Expertise Concentration Trap) in bug triage by dynamically balancing workloads between core veterans and long-tail developers.

By combining Latent Relationship Mining with an Atomicity-Aware Dual-View Routing mechanism, LANTERN structurally enfranchises isolated developers without inflating computational overhead.

---

## 📌 Important Notes for Artifact Evaluation

### 1. Toy Dataset & Raw Data Attribution
To facilitate rapid reproducibility checks for reviewers while protecting commercial and community data bandwidth, **the datasets included in this repository (`data/`) are heavily subsampled "Toy Datasets"**. They are structurally identical to the full datasets used in the paper and are fully sufficient to verify the pipeline's execution and model architecture.

For researchers wishing to reproduce the complete large-scale experiments or train from scratch, please refer to the original raw issue tracker data formalized by:
> *Mani, S., Sankaran, A., & Aralikatte, R. (2019). DeepTriage: Exploring the effectiveness of deep learning for bug triaging. In Proceedings of the ACM India Joint International Conference on Data Science and Management of Data (CoDS-COMAD).*

### 2. Note on Performance Metrics & Reproducibility
You may observe slightly higher or divergent performance metrics when running this repository compared to the figures reported in the official paper. This is expected behavior:
* **Robust Reporting:** The metrics presented in the paper represent the most robust and conservative baselines averaged across multiple runs, rather than cherry-picked peak values.
* **Code Iteration:** The codebase provided here has undergone minor engineering optimizations (e.g., streamlined data processing and routing efficiency) post-submission. 

To faithfully reflect the original contribution of the proposed methodology, we have chosen to retain the pre-optimization, conservative baselines in the manuscript.

---

## 🚀 Quick Start

**Requirements:** Python 3.8+ and PyTorch 1.12+.

```bash
# 1. Clone the repository
cd LANTERN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the evaluation pipeline (Default: Google Core toy dataset)
python run.py --dataset gc
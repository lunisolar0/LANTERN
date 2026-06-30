"""
Evaluation metrics for bug triage / developer recommendation.

Implements the standard ranking metrics used in the paper:
  - HR@K   (Hit Rate)
  - NDCG@K (Normalised Discounted Cumulative Gain)
  - MRR     (Mean Reciprocal Rank)
"""
import numpy as np
import torch


def compute_metrics(scores, pos_devs, k_list=(1, 3, 5, 10)):
    """
    Compute HR@K, NDCG@K, and MRR for a batch of predictions.

    Args:
        scores:   [N, D]  prediction scores for N query bugs against D developers
        pos_devs: [N]     ground-truth developer index for each bug
        k_list:   tuple of K values for HR and NDCG

    Returns:
        dict mapping metric name (e.g. ``"HR@5"``) → float value
    """
    N = scores.shape[0]
    max_k = max(k_list)

    _, topk_indices = scores.topk(max_k, dim=1)        # [N, max_k]
    topk = topk_indices.cpu().numpy()
    pos = pos_devs.cpu().numpy()

    metrics = {}

    # ── MRR ──
    mrr_sum = 0.0
    for i in range(N):
        rank = np.where(topk[i] == pos[i])[0]
        if len(rank) > 0:
            mrr_sum += 1.0 / (rank[0] + 1)
    metrics["MRR"] = mrr_sum / N

    # ── HR@K & NDCG@K ──
    for k in k_list:
        hr = 0
        ndcg = 0
        for i in range(N):
            p = pos[i]
            hits = (topk[i][:k] == p)
            if hits.any():
                hr += 1
                rank = np.where(hits)[0][0]
                ndcg += 1.0 / np.log2(rank + 2)       # IDCG = 1 for single relevant item
        metrics[f"HR@{k}"] = hr / N
        metrics[f"NDCG@{k}"] = ndcg / N

    return metrics


def evaluate_model(model, dataset, split="test", k_list=(1, 3, 5, 10)):
    """
    Evaluate a trained LANTERN model on a given data split.

    Args:
        model:   LANTERN model instance
        dataset: BipartiteDataset instance
        split:   ``"val"`` or ``"test"``
        k_list:  K values for HR and NDCG

    Returns:
        dict of metric name → float value
    """
    model.eval()
    with torch.no_grad():
        bug_exp, dev_fused = model()

        indices = dataset.val_idx if split == "val" else dataset.test_idx

        bug_h = bug_exp[indices]
        pos_devs = dataset.bug_owners[indices]
        pos_devs = torch.tensor(pos_devs, dtype=torch.long, device=bug_h.device)

        scores = bug_h @ dev_fused.T
        metrics = compute_metrics(scores, pos_devs, k_list)

    return metrics


def format_metrics(metrics):
    """Format a metrics dictionary as a compact one-line string."""
    return " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())

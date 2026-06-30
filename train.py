#!/usr/bin/env python3
"""
Training script for LANTERN.

Usage:
    python train.py gc     # train on Google Core
    python train.py mc     # train on Mozilla Core
    python train.py mf     # train on Mozilla Firefox

Uses full-batch BPR training: one forward pass through the entire graph per epoch,
followed by BPR loss computation over all training edges with a single backward step.
This approach is memory-efficient and avoids the need for graph retention across batches.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from lantern.config import (
    DATA_DIR, HIDDEN_DIM, NUM_LAYERS_EXP, NUM_LAYERS_IMP,
    TOP_K_AUG, TAU_MIN, EPOCHS, LR, L2_REG,
    EARLY_STOP_PATIENCE, DEVICE, TOP_K_EVAL,
)
from lantern.dataset import BipartiteDataset, set_seed
from lantern.model import LANTERN, build_augmented_adjacency
from lantern.utils import evaluate_model, format_metrics


def sample_negatives(pos_devs, num_devs):
    """
    Sample one negative developer per positive,
    ensuring the negative differs from the positive developer.
    """
    neg = torch.randint(0, num_devs, (len(pos_devs),), device=pos_devs.device)
    collision = (neg == pos_devs)
    while collision.any():
        neg[collision] = torch.randint(
            0, num_devs, (collision.sum().item(),), device=pos_devs.device
        )
        collision = (neg == pos_devs)
    return neg


def train(dataset_name):
    """Train LANTERN on a single dataset."""
    print(f"\n{'=' * 60}")
    print(f"  LANTERN — Training on {dataset_name}")
    print(f"{'=' * 60}")

    set_seed(42)

    # ── 1. Load and preprocess data ──
    dataset = BipartiteDataset(dataset_name)
    adj_sparse = dataset.get_train_adj()

    # ── 2. Build augmented adjacency (semantic voting, computed once) ──
    print(f"[{dataset_name}] Building augmented adjacency via semantic voting ...")
    aug_adj = build_augmented_adjacency(dataset, top_k=TOP_K_AUG, tau_min=TAU_MIN)

    # ── 3. Original developer degree (used by the adaptive gate) ──
    dev_degree = torch.sparse.sum(adj_sparse, dim=0).to_dense()

    # ── 4. Instantiate model ──
    model = LANTERN(
        num_bugs=dataset.num_bugs,
        num_devs=dataset.num_devs,
        bug_embeddings=dataset.bug_embeddings,
        aug_adj=aug_adj,
        dev_degree=dev_degree,
        hidden_dim=HIDDEN_DIM,
        num_layers_exp=NUM_LAYERS_EXP,
        num_layers_imp=NUM_LAYERS_IMP,
    )
    model.l2_reg = L2_REG
    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total  |  {n_trainable:,} trainable")

    # ── 5. Prepare training tensors ──
    train_bugs = torch.tensor(dataset.train_idx, dtype=torch.long, device=DEVICE)
    train_pos = torch.tensor(
        dataset.bug_owners[dataset.train_idx], dtype=torch.long, device=DEVICE
    )

    # ── 6. Optimizer ──
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── 7. Training loop ──
    best_val_mrr = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Full-graph forward pass (once per epoch)
        bug_exp, dev_fused = model()

        # Sample negatives and compute full-batch BPR loss
        train_neg = sample_negatives(train_pos, dataset.num_devs)

        bug_h = bug_exp[train_bugs]
        pos_h = dev_fused[train_pos]
        neg_h = dev_fused[train_neg]

        pos_score = (bug_h * pos_h).sum(dim=-1)
        neg_score = (bug_h * neg_h).sum(dim=-1)

        loss = -F.logsigmoid(pos_score - neg_score).mean()
        l2_reg = model.dev_embeddings.weight.pow(2).sum()
        loss = loss + model.l2_reg * l2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Validation ──
        val_metrics = evaluate_model(model, dataset, k_list=TOP_K_EVAL)
        val_mrr = val_metrics["MRR"]

        print(
            f"  Epoch {epoch:3d} | loss: {loss.item():.4f} | "
            f"val {format_metrics(val_metrics)}"
        )

        # Early stopping on MRR
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch}")
            break

    # ── 8. Final evaluation on test set ──
    print(f"\n  Restoring best model (epoch {best_epoch}, val MRR {best_val_mrr:.4f})")
    model.load_state_dict(best_state)
    model.to(DEVICE)

    test_metrics = evaluate_model(model, dataset, split="test", k_list=TOP_K_EVAL)
    print(f"  Test  {format_metrics(test_metrics)}")

    # ── 9. Save checkpoint ──
    save_dir = os.path.join(DATA_DIR, f"{dataset_name}_cache")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state, os.path.join(save_dir, "model.pt"))
    print(f"  Checkpoint saved to {save_dir}/model.pt")

    return test_metrics


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "gc"
    train(ds)

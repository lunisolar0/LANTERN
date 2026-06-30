"""
LANTERN model: Latent Relationship Mining + Atomicity-Aware Dual-View Routing.

Architecture (cf. Section 2 of the paper):
  1. Graph Augmentation (Sec. 2.1) — semantic voting + soft edge injection
  2. Explicit Bipartite Routing (Sec. 2.2, Eq. 4) — LightGCN-style propagation
  3. Developer-Projected Routing (Sec. 2.2, Eq. 5–6) — homogeneous dev-graph propagation
  4. Adaptive Load-Balancing Gate (Sec. 2.2, Eq. 7) — degree-aware fusion
  5. BPR loss (Sec. 2.3, Eq. 8) — Bayesian Personalized Ranking
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LANTERN(nn.Module):
    """
    LANTERN: end-to-end load-intervention framework for bug triage.

    Implements the two core modules described in the paper:
      (i)  Latent Relationship Mining — parameter-free semantic voting
           to repair atomicity-induced topology fragmentation.
      (ii) Atomicity-Aware Dual-View Routing — dual-path propagation with
           dynamic load-balancing gate to mitigate the Matthew Effect.

    Args:
        num_bugs:          number of bug report nodes  (N_b)
        num_devs:          number of developer nodes    (N_d)
        bug_embeddings:    pre-computed semantic vectors X_b  [N_b, d]
        aug_adj:           augmented adjacency Â = A + S⊙M     [N_b, N_d] (sparse COO)
        dev_degree:        original developer degree Σ_i A_ij  [N_d]
        hidden_dim:        hidden / developer embedding dimension
        num_layers_exp:    explicit propagation depth  (L_e)
        num_layers_imp:    implicit propagation depth  (L_i)
    """

    def __init__(
        self,
        num_bugs,
        num_devs,
        bug_embeddings,
        aug_adj,
        dev_degree,
        hidden_dim=64,
        num_layers_exp=3,
        num_layers_imp=2,
    ):
        super().__init__()

        self.num_bugs = num_bugs
        self.num_devs = num_devs
        self.hidden_dim = hidden_dim
        self.num_layers_exp = num_layers_exp
        self.num_layers_imp = num_layers_imp

        # ── Augmented adjacency Â = A + S⊙M  (frozen buffer) ──
        self.register_buffer("aug_adj_indices", aug_adj.indices())
        self.register_buffer("aug_adj_values", aug_adj.values())

        # ── Original developer degree for gate γ  (frozen, Eq. 7) ──
        self.register_buffer("dev_degree", dev_degree.float())

        # ── Frozen bug semantic vectors X_b ──
        self.register_buffer("bug_embeddings", bug_embeddings)
        embed_dim = bug_embeddings.shape[1]

        # ── Trainable developer ID embeddings H_d^(0) ──
        self.dev_embeddings = nn.Embedding(num_devs, hidden_dim)
        nn.init.xavier_uniform_(self.dev_embeddings.weight)

        # ── Trainable projection: R^d → R^hidden ──
        self.bug_proj = nn.Linear(embed_dim, hidden_dim, bias=False)

        # ── Gate scalars (trainable, Eq. 7): w, b ∈ R ──
        self.gate_w = nn.Parameter(torch.tensor(0.0))
        self.gate_b = nn.Parameter(torch.tensor(0.0))

        # ── Regularization strength λ (set externally during training) ──
        self.l2_reg = 0.0

    # ------------------------------------------------------------------
    #  Sparse adjacency reconstruction
    # ------------------------------------------------------------------

    @property
    def aug_adj(self):
        """Reconstruct Â from stored indices and values."""
        return torch.sparse_coo_tensor(
            self.aug_adj_indices,
            self.aug_adj_values,
            size=(self.num_bugs, self.num_devs),
            device=self.aug_adj_indices.device,
        ).coalesce()

    # ------------------------------------------------------------------
    #  Explicit Bipartite Routing  (Eq. 4)
    # ------------------------------------------------------------------

    def explicit_propagation(self):
        """
        LightGCN-style symmetric propagation over the augmented bipartite graph Â.

        H^(l+1)_exp = D̂^(u)^(-1/2) · Â^(u) · D̂^(u)^(-1/2) · H^(l)_exp

        where Â^(u) is the symmetric unified matrix built from Â.
        Layer-wise average pooling yields the final representations
        Z_exp (for bugs) and Z_exp (for developers).
        """
        aug_adj = self.aug_adj
        bug_h = self.bug_proj(self.bug_embeddings)   # X_b projected to hidden space
        dev_h = self.dev_embeddings.weight            # H_d^(0)

        bug_all = [bug_h]
        dev_all = [dev_h]

        # Degree normalization (computed once, frozen Â)
        dev_deg = torch.sparse.sum(aug_adj, dim=0).to_dense().clamp(min=1)
        bug_deg = torch.sparse.sum(aug_adj, dim=1).to_dense().clamp(min=1)
        dev_norm = dev_deg.pow(-0.5)
        bug_norm = bug_deg.pow(-0.5)

        for _ in range(self.num_layers_exp):
            # Bug ← Developer
            msg = aug_adj @ (dev_norm.unsqueeze(1) * dev_h)
            bug_h = bug_norm.unsqueeze(1) * msg
            # Developer ← Bug
            msg_t = aug_adj.t() @ (bug_norm.unsqueeze(1) * bug_h)
            dev_h = dev_norm.unsqueeze(1) * msg_t

            bug_all.append(bug_h)
            dev_all.append(dev_h)

        # Layer-wise average → Z_exp
        bug_exp = torch.stack(bug_all, dim=0).mean(dim=0)
        dev_exp = torch.stack(dev_all, dim=0).mean(dim=0)
        return bug_exp, dev_exp

    # ------------------------------------------------------------------
    #  Developer-Projected Routing  (Eq. 5–6)
    # ------------------------------------------------------------------

    def implicit_propagation(self):
        """
        Homogeneous developer-side propagation via the projection matrix

          R̂_dev = Âᵀ · D̂_b⁻¹ · Â

        H^(l+1)_d,imp = D̂_dev^(-1/2) · R̂_dev · D̂_dev^(-1/2) · H^(l)_d,imp

        Implemented with implicit matrix-vector products to avoid
        materialising the D×D matrix R̂_dev.
        """
        aug_adj = self.aug_adj
        dev_h = self.dev_embeddings.weight                      # H_d^(0)

        bug_deg = torch.sparse.sum(aug_adj, dim=1).to_dense().clamp(min=1)
        bug_norm_inv = bug_deg.pow(-1)                           # D̂_b⁻¹

        # D̂_dev = diag(R̂_dev · 1)
        adj_sum = torch.sparse.sum(aug_adj, dim=1).to_dense()
        dev_deg_r = torch.sparse.mm(
            aug_adj.t(),
            bug_norm_inv.unsqueeze(1) * adj_sum.unsqueeze(1),
        ).squeeze().clamp(min=1)
        dev_norm_r = dev_deg_r.pow(-0.5)

        dev_all = [dev_h]

        for _ in range(self.num_layers_imp):
            # msg_bug = Â · H   →  D̂_b⁻¹ · msg_bug   →  Âᵀ · (result)
            msg_bug = torch.sparse.mm(aug_adj, dev_h)
            msg_bug = bug_norm_inv.unsqueeze(1) * msg_bug
            dev_h = torch.sparse.mm(aug_adj.t(), msg_bug)
            dev_h = dev_norm_r.unsqueeze(1) * dev_h
            dev_all.append(dev_h)

        dev_imp = torch.stack(dev_all, dim=0).mean(dim=0)        # Z_d,imp
        return dev_imp

    # ------------------------------------------------------------------
    #  Dynamic Load-Balancing Gate  (Eq. 7)
    # ------------------------------------------------------------------

    def adaptive_fusion(self, dev_exp, dev_imp):
        """
        Degree-aware adaptive fusion:

          γⱼ = σ( w · log( Σ_i A_ij + 1 ) + b )
          z_dⱼ = γⱼ · z_dⱼ^exp + (1 − γⱼ) · z_dⱼ^imp

        The gate uses the ORIGINAL adjacency A (not augmented Â)
        so that the routing shift is governed strictly by true
        observed workloads.
        """
        gate_input = self.gate_w * torch.log(self.dev_degree + 1) + self.gate_b
        gamma = torch.sigmoid(gate_input)

        dev_fused = gamma.unsqueeze(1) * dev_exp + (1 - gamma).unsqueeze(1) * dev_imp
        return dev_fused

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def forward(self):
        """Full forward: explicit → implicit → gate fusion."""
        bug_exp, dev_exp = self.explicit_propagation()
        dev_imp = self.implicit_propagation()
        dev_fused = self.adaptive_fusion(dev_exp, dev_imp)
        return bug_exp, dev_fused

    # ------------------------------------------------------------------
    #  BPR Loss  (Eq. 8)
    # ------------------------------------------------------------------

    def bpr_loss(self, bug_idx, pos_dev, neg_dev, bug_exp, dev_fused):
        """
        Bayesian Personalized Ranking loss:

          L = −Σ_T log σ( ŷ_{i,j⁺} − ŷ_{i,j⁻} ) + λ ‖H_d^(0)‖₂²

        where ŷ = z_bᵢ · z_dⱼ is the dot-product score.
        L₂ penalty is applied exclusively to developer structural embeddings.
        """
        bug_h = bug_exp[bug_idx]
        pos_h = dev_fused[pos_dev]
        neg_h = dev_fused[neg_dev]

        pos_score = (bug_h * pos_h).sum(dim=1)
        neg_score = (bug_h * neg_h).sum(dim=1)

        loss = -F.logsigmoid(pos_score - neg_score).mean()

        # L₂ on developer embeddings only
        l2_reg = self.dev_embeddings.weight.pow(2).sum()
        loss = loss + self.l2_reg * l2_reg

        return loss


# ======================================================================
#  Graph Augmentation: Semantic Voting + Soft Edge Injection (Sec. 2.1)
# ======================================================================

def build_augmented_adjacency(dataset, top_k=5, tau_min=0.30):
    """
    Pre-compute the augmented adjacency matrix Â = A + S⊙M.

    Steps (cf. Section 2.1):
      1. Compute developer semantic centroids p_j (Eq. 1) via sparse aggregation.
      2. Compute cosine similarity S_ij between each bug x_i and each p_j.
      3. Mask existing edges (A_ij = 0 only), apply threshold τ_min.
      4. Retain Top-K candidates per bug to form binary mask M.
      5. Return Â = A + (S ⊙ M) as a sparse COO tensor.

    Implemented with chunked processing to avoid materialising the
    full [N_b × N_d] similarity matrix.
    """
    device = dataset.bug_embeddings.device
    adj_sparse = dataset.get_train_adj()
    bug_emb = dataset.bug_embeddings     # X_b  [N_b, d]  (L2-normalised)

    # ── Step 1: Developer centroids (Eq. 1) via sparse index_add ──
    dev_centroids = torch.zeros(dataset.num_devs, bug_emb.shape[1], device=device)
    dev_counts = torch.zeros(dataset.num_devs, device=device)
    dev_centroids.index_add_(
        0, adj_sparse.indices()[1], bug_emb[adj_sparse.indices()[0]]
    )
    dev_counts.index_add_(
        0, adj_sparse.indices()[1],
        torch.ones(adj_sparse._nnz(), device=device),
    )
    dev_counts = dev_counts.clamp(min=1)
    dev_centroids = dev_centroids / dev_counts.unsqueeze(1)
    dev_centroids = F.normalize(dev_centroids, dim=1)

    # ── Steps 2–4: Similarity, mask, threshold, Top-K (chunked) ──
    # Build existing-edge index for masking
    existing_edges = {}
    bug_idx_np = adj_sparse.indices()[0].cpu().numpy()
    dev_idx_np = adj_sparse.indices()[1].cpu().numpy()
    for bi, di in zip(bug_idx_np, dev_idx_np):
        existing_edges.setdefault(int(bi), set()).add(int(di))

    chunk_size = 4096
    num_bugs = dataset.num_bugs

    all_bug_idx = []
    all_dev_idx = []
    all_values = []

    for start in range(0, num_bugs, chunk_size):
        end = min(start + chunk_size, num_bugs)
        chunk_emb = bug_emb[start:end]

        sim_chunk = chunk_emb @ dev_centroids.T             # S_ij for this chunk

        # Mask observed edges
        for i in range(sim_chunk.shape[0]):
            bug_id = start + i
            if bug_id in existing_edges:
                for dev_id in existing_edges[bug_id]:
                    sim_chunk[i, dev_id] = -1e9

        sim_chunk[sim_chunk < tau_min] = -1e9               # threshold

        _, topk_idx = sim_chunk.topk(top_k, dim=1)           # Top-K
        topk_val = sim_chunk.gather(1, topk_idx)

        valid_mask = topk_val > 0
        bug_offset = torch.arange(start, end, device=device).unsqueeze(1).expand(-1, top_k)
        all_bug_idx.append(bug_offset[valid_mask].cpu())
        all_dev_idx.append(topk_idx[valid_mask].cpu())
        all_values.append(topk_val[valid_mask].cpu())

    # ── Step 5: Â = A + S⊙M ──
    soft_bug = torch.cat(all_bug_idx).to(device)
    soft_dev = torch.cat(all_dev_idx).to(device)
    soft_val = torch.cat(all_values).to(device)

    orig_indices = adj_sparse.indices()
    orig_values = adj_sparse.values()

    all_bug = torch.cat([orig_indices[0], soft_bug])
    all_dev = torch.cat([orig_indices[1], soft_dev])
    all_val = torch.cat([orig_values, soft_val])

    aug_adj = torch.sparse_coo_tensor(
        torch.stack([all_bug, all_dev]), all_val,
        size=(dataset.num_bugs, dataset.num_devs),
        device=device,
    ).coalesce()

    print(
        f"  Augmented edges: {aug_adj._nnz()} "
        f"(original: {adj_sparse._nnz()}, +{len(soft_val)} soft)"
    )

    return aug_adj

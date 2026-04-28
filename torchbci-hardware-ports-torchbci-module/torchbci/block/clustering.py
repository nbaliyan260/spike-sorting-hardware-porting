import torch
from .block import Block
from typing import List, Tuple


class Clustering(Block):
    """Base class for all clustering blocks.

    Args:
        Block (torch.nn.Module): Inherits from the Block class.
    """    
    def __init__(self):
        super().__init__()



class SimpleOnlineKMeansClustering(Clustering):
    """
    Streaming clustering with simple online k-means updates.

    State:
      - self.centroids: (k, d) tensor of running centroids
      - self.counts:    (k,) number of points seen per cluster (for info only)
      - self.clusters:      list[k] of lists, each inner list stores feature tensors (detached, on CPU)
      - self.clusters_meta: list[k] of lists, each inner list stores meta tensors (detached, on CPU)
    """
    def __init__(self,
                 n_clusters: int,
                 cluster_feature_size: int):
        super().__init__()
        self.n_clusters = n_clusters
        self.cluster_feature_size = cluster_feature_size

        # Online update hyperparams (tweak on the instance if needed)
        self.lr: float = 0.50     # EMA step size for centroid updates
        self.minibatch_merge: bool = True  # use batch means to update once per cluster per forward

        # Internal state tensors (lazy-init on first call to forward to match dtype/device)
        self.centroids: torch.Tensor = None
        self.counts: torch.Tensor = None  # float counts for info; EMA doesn’t require it

        self._initialize_clusters()

    def _initialize_clusters(self):
        self.clusters = [[] for _ in range(self.n_clusters)]
        self.clusters_meta = [[] for _ in range(self.n_clusters)]

    def _lazy_tensor_init(self, x: torch.Tensor):
        """Create centroid/count tensors on the same device/dtype as x."""
        if self.centroids is None:
            k, d = self.n_clusters, self.cluster_feature_size
            if d != x.shape[-1]:
                raise ValueError(f"cluster_feature_size={d} but got features with dim={x.shape[-1]}")
            self.centroids = torch.zeros(k, d, device=x.device, dtype=x.dtype)
            self.counts = torch.zeros(k, device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def forward(self,
                aligned_features: torch.Tensor,
                aligned_meta: torch.Tensor
                ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """
        Args
        ----
        aligned_features: (N, D) tensor of features
        aligned_meta:     (N, 2) tensor of (channel, time) or similar identifiers

        Side effects
        ------------
        - Updates running centroids in-place using EMA
        - Appends (detached CPU) copies of features and meta into per-cluster buckets

        Returns
        -------
        cluster_ids: (N,) long tensor of assigned cluster indices
        centroids:   (K, D) tensor of current centroids (same device/dtype as inputs)
        """
        if aligned_features.ndim != 2:
            raise ValueError(f"aligned_features must be (N, D), got shape {tuple(aligned_features.shape)}")
        if aligned_meta.ndim != 2 or aligned_meta.shape[0] != aligned_features.shape[0]:
            raise ValueError("aligned_meta must be (N, 2) with same N as features")

        N, D = aligned_features.shape
        if N == 0:
            # Nothing to do; return current state (or zeros if never initialized)
            self._lazy_tensor_init(torch.zeros(1, self.cluster_feature_size, device=aligned_features.device,
                                               dtype=aligned_features.dtype))
            return torch.empty(0, dtype=torch.long, device=aligned_features.device), self.centroids

        self._lazy_tensor_init(aligned_features)

        x = aligned_features
        m = aligned_meta

        # --- Cold-start seeding: fill any empty cluster with early points
        empty_mask = (self.counts == 0)
        num_empty = int(empty_mask.sum().item())
        cluster_ids = torch.empty(N, dtype=torch.long, device=x.device)

        if num_empty > 0:
            # Indices of clusters that need a seed
            empty_slots = torch.nonzero(empty_mask, as_tuple=False).flatten()
            seed_count = min(num_empty, N)
            # Use the first `seed_count` points to seed empty clusters
            self.centroids[empty_slots[:seed_count]] = x[:seed_count]
            self.counts[empty_slots[:seed_count]] = 1.0

            # Route those seeded points
            seeded_ids = empty_slots[:seed_count]
            cluster_ids[:seed_count] = seeded_ids

            # Store to buckets (detach to CPU so we don't retain graph or GPU mem)
            for idx_pt, cid in enumerate(seeded_ids.tolist()):
                self.clusters[cid].append(x[idx_pt].detach().cpu())
                self.clusters_meta[cid].append(m[idx_pt].detach().cpu().tolist())

            # If we still have leftover points, assign them normally below
            start = seed_count
        else:
            start = 0

        # --- Assign remaining points to nearest centroids (vectorized)
        if start < N:
            # Squared Euclidean distance to centroids
            # (N', K, D) -> (N', K)
            diffs = x[start:, None, :] - self.centroids[None, :, :]
            d2 = (diffs * diffs).sum(dim=-1)
            nearest = torch.argmin(d2, dim=1)  # (N',)

            cluster_ids[start:] = nearest

            # Optionally update centroids using batch means per cluster (more stable than per-point EMA)
            if self.minibatch_merge:
                for cid in nearest.unique().tolist():
                    mask = (nearest == cid)
                    if mask.any():
                        x_c = x[start:][mask]
                        # Batch-mean update with EMA
                        self.centroids[cid] = (1.0 - self.lr) * self.centroids[cid] + self.lr * x_c.mean(dim=0)
                        self.counts[cid] += float(mask.sum().item())

                        # Persist features/meta to Python lists
                        # (detach to CPU to avoid GPU growth and autograd references)
                        xs = x_c.detach().cpu()
                        ms = m[start:][mask].detach().cpu()
                        self.clusters[cid].extend(xs)
                        self.clusters_meta[cid].extend(ms.tolist())
            else:
                # Per-point EMA (slightly noisier but even simpler)
                for i in range(start, N):
                    cid = int(nearest[i - start].item())
                    self.centroids[cid] = (1.0 - self.lr) * self.centroids[cid] + self.lr * x[i]
                    self.counts[cid] += 1.0
                    self.clusters[cid].append(x[i].detach().cpu())
                    self.clusters_meta[cid].append(m[i].detach().cpu().tolist())

        return self.clusters, self.centroids, self.clusters_meta
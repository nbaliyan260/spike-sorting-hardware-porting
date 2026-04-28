import warnings

import numpy as np
import torch
from scipy.cluster.hierarchy import ClusterWarning, cut_tree, linkage
from scipy.spatial.distance import squareform
from sklearn import decomposition
#from isosplit6 import isosplit6

from torchbci.block.mountainsort5_blocks.mountainsort5block import MountainSort5Block

warnings.filterwarnings("ignore", category=ClusterWarning)


class Isosplit6ClusteringBlock(MountainSort5Block):
    """
    Cluster PCA features using the isosplit6 subdivision method.
    """

    def __init__(self, params):
        super().__init__()
        self.npca_per_subdivision = params.npca_per_subdivision

    def run_block(self, batch):
        features = batch.features
        assert features is not None

        labels = _cluster(features, npca_per_subdivision=self.npca_per_subdivision)
        batch.labels = labels.to(features.device)
        return batch


def _cluster(features: torch.Tensor, *, npca_per_subdivision: int) -> torch.Tensor:
    if features.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.int64)

    X = features.detach().cpu().numpy().astype(np.float32, copy=False)
    ret_labels = _split_clusters(X, npca_per_subdivision=npca_per_subdivision)
    return torch.tensor(ret_labels, dtype=torch.int64)


def _split_clusters(X: np.ndarray, npca_per_subdivision: int):
    from isosplit6 import isosplit6
    num_events = X.shape[0]
    if num_events == 0:
        return np.zeros((0,), dtype=np.int32)
    if num_events == 1:
        return np.ones((1,), dtype=np.int32)

    max_num_centroids = 100
    if num_events > max_num_centroids:
        kmeans = decomposition.MiniBatchKMeans(
            n_clusters=max_num_centroids,
            random_state=0,
            batch_size=max_num_centroids * 10,
            n_init="auto",
        ).fit(X)
        centroids = kmeans.cluster_centers_
        assignments = kmeans.labels_
    else:
        centroids = X
        assignments = np.arange(num_events)

    Z = linkage(centroids, method="ward")
    cluster_inds = cut_tree(Z, n_clusters=np.arange(max_num_centroids) + 1)

    best_aa = float("inf")
    best_labels = None

    for ii in range(min(max_num_centroids, num_events)):
        labels0 = cluster_inds[:, ii]
        labels = np.zeros(num_events, dtype=np.int32)
        k_offset = 1

        for i in range(ii + 1):
            inds = np.where(labels0 == i)[0]
            if len(inds) == 0:
                continue

            inds2 = np.where(np.isin(assignments, inds))[0]
            X_sub = X[inds2]

            if len(inds2) < 20:
                labels_sub = np.ones(len(inds2), dtype=np.int32)
            else:
                labels_sub = isosplit6(X_sub.T)

            labels[inds2] = labels_sub + k_offset - 1
            k_offset += labels_sub.max()

        K = labels.max()
        aa = K + 0.1 * ii
        if aa < best_aa:
            best_aa = aa
            best_labels = labels

    assert best_labels is not None
    ret_labels = best_labels
    return ret_labels
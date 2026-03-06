"""Graph construction from superpixel features.

Builds k-NN graphs where nodes are superpixels and edges represent
spatial adjacency and spectral similarity.
"""

import numpy as np
import torch
from torch_geometric.data import Data


def build_adjacency_from_segments(segments: np.ndarray) -> set[tuple[int, int]]:
    """Extract adjacency edges from a superpixel segmentation map.

    Two superpixels are adjacent if they share at least one boundary pixel.

    Args:
        segments: Label array of shape (H, W).

    Returns:
        Set of (i, j) tuples representing undirected edges.
    """
    edges = set()
    h, w = segments.shape

    for i in range(h):
        for j in range(w):
            current = segments[i, j]
            # Check right neighbor
            if j + 1 < w and segments[i, j + 1] != current:
                edge = (min(current, segments[i, j + 1]), max(current, segments[i, j + 1]))
                edges.add(edge)
            # Check bottom neighbor
            if i + 1 < h and segments[i + 1, j] != current:
                edge = (min(current, segments[i + 1, j]), max(current, segments[i + 1, j]))
                edges.add(edge)

    return edges


def build_knn_graph(features: np.ndarray, k: int = 8) -> np.ndarray:
    """Build k-nearest-neighbor graph from feature vectors.

    Args:
        features: Node feature matrix of shape (N, D).
        k: Number of nearest neighbors per node.

    Returns:
        Edge index array of shape (2, num_edges).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(features)), metric="euclidean")
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    src_nodes = []
    dst_nodes = []

    for node_id in range(len(features)):
        for neighbor_id in indices[node_id, 1:]:  # skip self
            src_nodes.append(node_id)
            dst_nodes.append(neighbor_id)

    edge_index = np.array([src_nodes, dst_nodes], dtype=np.int64)
    return edge_index


def build_combined_graph(
    features: np.ndarray,
    segments: np.ndarray,
    k: int = 8,
    spatial_weight: float = 0.5,
) -> np.ndarray:
    """Build graph combining spatial adjacency and feature-based k-NN edges.

    Args:
        features: Node feature matrix (N, D).
        segments: Superpixel label array (H, W).
        k: Number of nearest neighbors.
        spatial_weight: Not used directly; both edge types are included.

    Returns:
        Edge index array (2, num_edges) with deduplicated edges.
    """
    # Spatial adjacency edges
    spatial_edges = build_adjacency_from_segments(segments)

    # Feature-based k-NN edges
    knn_edge_index = build_knn_graph(features, k)

    # Combine
    all_edges = set()
    for i, j in spatial_edges:
        all_edges.add((i, j))
        all_edges.add((j, i))  # undirected

    for idx in range(knn_edge_index.shape[1]):
        i, j = knn_edge_index[0, idx], knn_edge_index[1, idx]
        all_edges.add((i, j))

    if not all_edges:
        return np.zeros((2, 0), dtype=np.int64)

    edges = np.array(list(all_edges), dtype=np.int64).T
    return edges


def create_pyg_data(
    features: np.ndarray,
    edge_index: np.ndarray,
    labels: np.ndarray | None = None,
) -> Data:
    """Create a PyTorch Geometric Data object.

    Args:
        features: Node features (N, D).
        edge_index: Edge index (2, E).
        labels: Optional node labels (N,).

    Returns:
        PyG Data object ready for GNN training.
    """
    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )

    if labels is not None:
        data.y = torch.tensor(labels, dtype=torch.long)

    return data

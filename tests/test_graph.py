"""Tests for graph construction module."""

import numpy as np
import pytest
from src.graph_construction import build_adjacency_from_segments, build_knn_graph, create_pyg_data


class TestGraphConstruction:
    def test_adjacency_from_segments(self):
        segments = np.array([[0, 0, 1], [0, 2, 1], [2, 2, 1]])
        edges = build_adjacency_from_segments(segments)
        assert len(edges) > 0
        assert (0, 1) in edges or (1, 0) in edges

    def test_knn_graph(self):
        features = np.random.randn(10, 5).astype(np.float32)
        edge_index = build_knn_graph(features, k=3)
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0

    def test_pyg_data_creation(self):
        features = np.random.randn(10, 5).astype(np.float32)
        edge_index = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        data = create_pyg_data(features, edge_index, labels)
        assert data.x.shape == (10, 5)
        assert data.edge_index.shape == (2, 3)
        assert data.y.shape == (10,)


class TestPreprocessing:
    def test_spectral_indices(self):
        from src.preprocessing import compute_spectral_indices

        bands = {
            "B04": np.random.rand(64, 64).astype(np.float32) * 3000,
            "B03": np.random.rand(64, 64).astype(np.float32) * 3000,
            "B08": np.random.rand(64, 64).astype(np.float32) * 5000,
            "B11": np.random.rand(64, 64).astype(np.float32) * 3000,
        }
        indices = compute_spectral_indices(bands)
        assert "NDVI" in indices
        assert "NDWI" in indices
        assert "NDBI" in indices
        assert indices["NDVI"].shape == (64, 64)
        # NDVI should be in [-1, 1]
        assert indices["NDVI"].min() >= -1.0
        assert indices["NDVI"].max() <= 1.0

    def test_tile_image(self):
        from src.preprocessing import tile_image

        image = np.random.rand(512, 512, 3)
        tiles = tile_image(image, tile_size=256, overlap=0)
        assert len(tiles) == 4  # 2x2 grid
        assert tiles[0][0].shape == (256, 256, 3)

    def test_superpixel_features(self):
        from src.preprocessing import extract_superpixel_features

        image = np.random.rand(64, 64, 4).astype(np.float32)
        segments = np.zeros((64, 64), dtype=np.int32)
        segments[32:, :] = 1
        segments[:, 32:] += 2

        features = extract_superpixel_features(image, segments)
        assert features.shape == (4, 8)  # 4 segments, 2*4 features

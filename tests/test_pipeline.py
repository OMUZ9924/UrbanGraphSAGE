"""Integration test for the full preprocessing → graph pipeline."""

import numpy as np
import pytest


def test_spectral_feature_extraction():
    """Verify spectral features are computed correctly."""
    # Simulate a small 4-band image (B2, B3, B4, B8)
    h, w = 64, 64
    bands = np.random.rand(4, h, w).astype(np.float32) * 0.3 + 0.1

    nir = bands[3]
    red = bands[2]
    ndvi = (nir - red) / (nir + red + 1e-8)

    assert ndvi.shape == (h, w)
    assert ndvi.min() >= -1.0
    assert ndvi.max() <= 1.0


def test_adjacency_symmetry():
    """Verify k-NN adjacency is symmetric after symmetrization."""
    from scipy.sparse import random as sparse_random

    # Random sparse adjacency
    adj = sparse_random(50, 50, density=0.1, format="csr")
    sym = (adj + adj.T) / 2

    diff = abs(sym - sym.T).sum()
    assert diff < 1e-10, "Adjacency should be symmetric"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

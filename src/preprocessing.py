"""Sentinel-2 preprocessing pipeline.

Loads raw Sentinel-2 L2A tiles, applies cloud masking, computes spectral indices,
and generates SLIC superpixels for graph construction.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_spectral_indices(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute NDVI, NDWI, and NDBI from Sentinel-2 bands.

    Args:
        bands: Dictionary mapping band names (B02, B03, B04, B08, B11) to arrays.

    Returns:
        Dictionary with computed indices.
    """
    eps = 1e-8
    nir = bands["B08"].astype(np.float32)
    red = bands["B04"].astype(np.float32)
    green = bands["B03"].astype(np.float32)
    swir = bands["B11"].astype(np.float32)

    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndbi = (swir - nir) / (swir + nir + eps)

    return {"NDVI": ndvi, "NDWI": ndwi, "NDBI": ndbi}


def generate_superpixels(image: np.ndarray, n_segments: int = 500, compactness: float = 20.0) -> np.ndarray:
    """Generate SLIC superpixels from a multispectral image.

    Args:
        image: Array of shape (H, W, C) with normalized spectral bands.
        n_segments: Target number of superpixels.
        compactness: Balance between color and spatial proximity.

    Returns:
        Label array of shape (H, W) with superpixel assignments.
    """
    from skimage.segmentation import slic

    segments = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=-1,
    )
    logger.info(f"Generated {segments.max() + 1} superpixels")
    return segments


def extract_superpixel_features(image: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Extract mean and std features per superpixel.

    Args:
        image: Array of shape (H, W, C).
        segments: Superpixel label array of shape (H, W).

    Returns:
        Feature matrix of shape (n_superpixels, 2*C) with mean and std per channel.
    """
    n_segments = segments.max() + 1
    n_channels = image.shape[2]
    features = np.zeros((n_segments, 2 * n_channels), dtype=np.float32)

    for seg_id in range(n_segments):
        mask = segments == seg_id
        pixels = image[mask]  # (N_pixels, C)
        if len(pixels) == 0:
            continue
        features[seg_id, :n_channels] = pixels.mean(axis=0)
        features[seg_id, n_channels:] = pixels.std(axis=0)

    return features


def create_cloud_mask(scl_band: np.ndarray) -> np.ndarray:
    """Create binary cloud mask from Sentinel-2 Scene Classification Layer.

    SCL classes 8 (cloud medium probability), 9 (cloud high probability),
    and 10 (thin cirrus) are flagged.

    Args:
        scl_band: Scene Classification Layer array.

    Returns:
        Binary mask where True = cloudy pixel.
    """
    cloud_classes = {8, 9, 10}
    return np.isin(scl_band, list(cloud_classes))


def tile_image(image: np.ndarray, tile_size: int = 256, overlap: int = 0) -> list[tuple[np.ndarray, int, int]]:
    """Split image into tiles.

    Args:
        image: Array of shape (H, W, C).
        tile_size: Size of each square tile.
        overlap: Number of overlapping pixels between tiles.

    Returns:
        List of (tile_array, row_offset, col_offset) tuples.
    """
    h, w = image.shape[:2]
    step = tile_size - overlap
    tiles = []

    for i in range(0, h - tile_size + 1, step):
        for j in range(0, w - tile_size + 1, step):
            tile = image[i : i + tile_size, j : j + tile_size]
            tiles.append((tile, i, j))

    logger.info(f"Created {len(tiles)} tiles of size {tile_size}x{tile_size}")
    return tiles


def preprocess_pipeline(input_dir: str, output_dir: str, n_segments: int = 500, tile_size: int = 256):
    """Run the full preprocessing pipeline.

    1. Load Sentinel-2 bands
    2. Apply cloud masking
    3. Compute spectral indices
    4. Tile the image
    5. Generate superpixels per tile
    6. Extract features per superpixel
    7. Save processed data
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing tiles from {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Superpixels per tile: {n_segments}, Tile size: {tile_size}")

    # NOTE: In production, this loads actual .tif files via rasterio
    # For demonstration, we show the pipeline structure
    logger.info("Pipeline steps: load → cloud_mask → indices → tile → superpixels → features → save")
    logger.info("See notebooks/01_eda.ipynb for a walkthrough with sample data")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Sentinel-2 imagery")
    parser.add_argument("--input", type=str, required=True, help="Input directory with Sentinel-2 bands")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--n-segments", type=int, default=500, help="Superpixels per tile")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size in pixels")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    preprocess_pipeline(args.input, args.output, args.n_segments, args.tile_size)


if __name__ == "__main__":
    main()

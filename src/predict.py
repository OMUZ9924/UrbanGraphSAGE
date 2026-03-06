"""Inference and evaluation script."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from .model import GraphSAGEClassifier
from .utils import compute_metrics

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict(model, data, device):
    """Run inference on graph data."""
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    return pred.cpu().numpy(), torch.exp(out).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Run UrbanGraphSAGE inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    logger.info(f"Using device: {device}")


if __name__ == "__main__":
    main()

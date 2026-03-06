"""Training script for UrbanGraphSAGE."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

from .model import GraphSAGEClassifier
from .utils import set_seed, compute_metrics, EarlyStopping

logger = logging.getLogger(__name__)


def train_epoch(model, loader, optimizer, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index)
        loss = F.nll_loss(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_nodes
        pred = out.argmax(dim=1)
        total_correct += (pred == batch.y).sum().item()
        total_nodes += batch.num_nodes

    return total_loss / total_nodes, total_correct / total_nodes


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        loss = F.nll_loss(out, batch.y)

        total_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes

        pred = out.argmax(dim=1)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / total_nodes

    return metrics


def train(config: dict):
    """Main training function."""
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Model
    model = GraphSAGEClassifier(
        in_channels=config["model"]["in_channels"],
        hidden_channels=config["model"]["hidden_channels"],
        num_classes=config["model"]["num_classes"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config["training"].get("patience", 15),
        min_delta=config["training"].get("min_delta", 1e-4),
    )

    # Training loop
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    logger.info(f"Starting training for {config['training']['epochs']} epochs")

    for epoch in range(1, config["training"]["epochs"] + 1):
        start = time.time()

        # NOTE: DataLoader would be populated with actual graph data
        # train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        # val_metrics = evaluate(model, val_loader, device)

        scheduler.step()
        elapsed = time.time() - start

        # Logging would happen here:
        # logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
        #             f"Val F1: {val_metrics['f1']:.4f} | Time: {elapsed:.1f}s")

    logger.info("Training complete")


def main():
    parser = argparse.ArgumentParser(description="Train UrbanGraphSAGE")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()

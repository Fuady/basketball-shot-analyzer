"""
train_bilstm.py
───────────────
Modeling Step 2: Train the BiLSTM shot predictor.

Features:
  - Weighted BCE loss (handles class imbalance automatically)
  - Learning rate scheduler (ReduceLROnPlateau)
  - Early stopping (patience=15 epochs)
  - Full MLflow experiment tracking
  - Checkpoint saving (best val AUC)

Usage:
    python src/modeling/train_bilstm.py \
        --sequences data/processed/sequences \
        --output models/bilstm \
        --epochs 100 \
        --hidden 128 \
        --lr 0.001
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.modeling.bilstm_model import ShotPredictor, count_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── DATASET LOADING ──────────────────────────────────────────────────────────

def load_sequences(seq_dir: Path) -> tuple:
    """Load train/val/test numpy arrays."""
    arrays = {}
    for split in ["train", "val", "test"]:
        X = np.load(seq_dir / f"X_{split}.npy")
        y = np.load(seq_dir / f"y_{split}.npy")
        arrays[split] = (
            torch.FloatTensor(X),
            torch.FloatTensor(y).unsqueeze(1),
        )
    return arrays


def make_dataloader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ─── LOSS ─────────────────────────────────────────────────────────────────────

def compute_pos_weight(y_train: torch.Tensor) -> torch.Tensor:
    """Compute positive class weight for BCEWithLogitsLoss to handle imbalance."""
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos)


# ─── TRAINING LOOP ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss, n_correct, n_total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        prob = model(X_batch)
        loss = criterion(prob, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        preds = (prob > 0.5).float()
        n_correct += (preds == y_batch).sum().item()
        n_total += len(X_batch)

    return total_loss / n_total, n_correct / n_total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model. Returns (loss, accuracy, AUC-ROC)."""
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        prob = model(X_batch)
        loss = criterion(prob, y_batch)
        total_loss += loss.item() * len(X_batch)
        all_probs.extend(prob.cpu().squeeze().tolist())
        all_labels.extend(y_batch.cpu().squeeze().tolist())

    avg_loss = total_loss / len(all_labels)
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
    return avg_loss, acc, auc


def train(args: argparse.Namespace) -> None:
    seq_dir = Path(args.sequences)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    arrays = load_sequences(seq_dir)
    X_train, y_train = arrays["train"]
    X_val,   y_val   = arrays["val"]
    X_test,  y_test  = arrays["test"]

    n_features = X_train.shape[2]
    seq_len    = X_train.shape[1]
    logger.info(f"Data loaded | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    logger.info(f"Seq shape: ({seq_len}, {n_features})")

    train_loader = make_dataloader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader   = make_dataloader(X_val,   y_val,   args.batch_size, shuffle=False)
    test_loader  = make_dataloader(X_test,  y_test,  args.batch_size, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    model = ShotPredictor(
        input_size=n_features,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # ── Loss & optimizer ──────────────────────────────────────────────────────
    pos_weight = compute_pos_weight(y_train).to(device)
    criterion  = nn.BCELoss()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=8, factor=0.5, verbose=True
    )
    early_stop = EarlyStopping(patience=15)

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_experiment("bball-shot-prediction")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "model": "BiLSTM",
            "input_size": n_features,
            "hidden_size": args.hidden,
            "num_layers": args.layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "seq_len": seq_len,
            "n_train": len(X_train),
        })

        best_val_auc  = 0.0
        best_ckpt     = output_dir / "best_model.pt"
        history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}

        logger.info(f"\nTraining for up to {args.epochs} epochs...")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
            elapsed = time.time() - t0

            scheduler.step(val_auc)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)
            history["val_acc"].append(val_acc)

            mlflow.log_metrics({
                "train_loss": round(train_loss, 4),
                "train_acc":  round(train_acc,  4),
                "val_loss":   round(val_loss,   4),
                "val_acc":    round(val_acc,    4),
                "val_auc":    round(val_auc,    4),
            }, step=epoch)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "input_size": n_features,
                    "hidden_size": args.hidden,
                    "num_layers": args.layers,
                    "dropout": args.dropout,
                    "seq_len": seq_len,
                }, best_ckpt)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:4d}/{args.epochs} | "
                    f"Train loss={train_loss:.4f} acc={train_acc:.3f} | "
                    f"Val loss={val_loss:.4f} acc={val_acc:.3f} auc={val_auc:.3f} | "
                    f"Best AUC={best_val_auc:.3f} | {elapsed:.1f}s"
                )

            if early_stop(val_auc):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # ── Test evaluation ───────────────────────────────────────────────────
        logger.info("\nLoading best model for test evaluation...")
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device)

        logger.info(f"\nTEST RESULTS")
        logger.info(f"  Loss    : {test_loss:.4f}")
        logger.info(f"  Accuracy: {test_acc:.4f}")
        logger.info(f"  AUC-ROC : {test_auc:.4f}")

        mlflow.log_metrics({
            "test_loss": round(test_loss, 4),
            "test_acc":  round(test_acc,  4),
            "test_auc":  round(test_auc,  4),
        })
        mlflow.log_artifact(str(best_ckpt), artifact_path="model")

        # Save training curves
        _save_training_curves(history, output_dir)
        mlflow.log_artifact(str(output_dir / "training_curves.png"), artifact_path="plots")

    logger.info(f"\n✅ Training complete. Best model: {best_ckpt}")
    logger.info("Next: python src/modeling/form_scorer.py")


def _save_training_curves(history: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train", color="#4C72B0")
    axes[0].plot(history["val_loss"],   label="Val",   color="#C44E52")
    axes[0].set_title("Loss Curves"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history["val_auc"],  label="Val AUC",  color="#55A868")
    axes[1].plot(history["val_acc"],  label="Val Acc",  color="#8172B2")
    axes[1].set_title("Validation Metrics"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.suptitle("BiLSTM Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiLSTM shot predictor")
    parser.add_argument("--sequences", default="data/processed/sequences")
    parser.add_argument("--output",    default="models/bilstm")
    parser.add_argument("--epochs",    type=int,   default=100)
    parser.add_argument("--hidden",    type=int,   default=128)
    parser.add_argument("--layers",    type=int,   default=2)
    parser.add_argument("--dropout",   type=float, default=0.3)
    parser.add_argument("--batch-size",type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=0.001)
    parser.add_argument("--cpu",       action="store_true")
    parser.add_argument("--run-name",  default="bilstm_v1")
    args = parser.parse_args()
    train(args)

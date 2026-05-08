"""
evaluate.py
───────────
Full evaluation of the BiLSTM shot predictor and form scorer.

Outputs:
  - BiLSTM: accuracy, AUC-ROC, confusion matrix, calibration curve
  - Form scorer: MAE, R², feature importance, SHAP
  - Saved plots → docs/evaluation/

Usage:
    python src/modeling/evaluate.py \
        --bilstm  models/bilstm/best_model.pt \
        --form    models/form_scorer.pkl \
        --seqs    data/processed/sequences \
        --release data/processed/features/release_features.csv
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, r2_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.modeling.bilstm_model import ShotPredictor
from src.modeling.form_scorer import FORM_FEATURES, create_form_score_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


# ─── BILSTM EVALUATION ────────────────────────────────────────────────────────

def evaluate_bilstm(ckpt_path: Path, seq_dir: Path, output_dir: Path) -> dict:
    if not ckpt_path.exists():
        logger.warning(f"BiLSTM checkpoint not found: {ckpt_path}")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = ShotPredictor(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X_test = torch.FloatTensor(np.load(seq_dir / "X_test.npy")).to(device)
    y_test = np.load(seq_dir / "y_test.npy")

    with torch.no_grad():
        probs = model(X_test).cpu().squeeze().numpy()

    preds = (probs > 0.5).astype(int)
    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, probs)

    logger.info("\nBiLSTM TEST RESULTS")
    logger.info(f"  Accuracy : {acc:.4f}")
    logger.info(f"  AUC-ROC  : {auc:.4f}")
    logger.info("\n" + classification_report(y_test, preds, target_names=["Miss", "Make"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Miss","Make"], yticklabels=["Miss","Make"], ax=axes[0])
    axes[0].set_title("Confusion Matrix"); axes[0].set_ylabel("True"); axes[0].set_xlabel("Pred")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    axes[1].plot(fpr, tpr, color="#2ecc71", lw=2, label=f"BiLSTM (AUC={auc:.3f})")
    axes[1].plot([0,1],[0,1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve — Shot Prediction"); axes[1].legend()

    plt.suptitle("BiLSTM Shot Predictor — Test Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "bilstm_evaluation.png"
    plt.savefig(out, dpi=120); plt.close()
    logger.info(f"Saved: {out}")

    # Calibration
    _plot_calibration(y_test, probs, output_dir)

    return {"bilstm_accuracy": round(acc, 4), "bilstm_auc": round(auc, 4)}


def _plot_calibration(y_true, probs, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10)
    ax.plot(mean_pred, frac_pos, "s-", color="#4C72B0", label="BiLSTM")
    ax.plot([0,1],[0,1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(); plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=120); plt.close()
    logger.info("Saved: calibration_curve.png")


# ─── FORM SCORER EVALUATION ───────────────────────────────────────────────────

def evaluate_form_scorer(
    model_path: Path,
    release_path: Path,
    output_dir: Path,
) -> dict:
    if not model_path.exists() or not release_path.exists():
        logger.warning("Form scorer or release features not found.")
        return {}

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    pipeline = artifact["pipeline"]
    feat_cols = artifact["feature_cols"]

    release_df = pd.read_csv(release_path)
    available = [f for f in feat_cols if f in release_df.columns]
    X = release_df[available].fillna(0)
    y = create_form_score_labels(release_df)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    logger.info(f"\nFORM SCORER TEST RESULTS")
    logger.info(f"  MAE : {mae:.2f} points")
    logger.info(f"  R²  : {r2:.4f}")

    # Scatter: predicted vs actual
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.4, color="#4C72B0", s=20)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    axes[0].set_xlabel("True Form Score"); axes[0].set_ylabel("Predicted Form Score")
    axes[0].set_title(f"Form Scorer: Predicted vs Actual (MAE={mae:.2f})")
    axes[0].legend()

    # Feature importance
    clf = pipeline.named_steps["reg"]
    importance = clf.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    axes[1].barh([available[i] for i in sorted_idx[:10]], importance[sorted_idx[:10]],
                 color="#55A868", edgecolor="white")
    axes[1].set_title("Top Feature Importances (Form Scorer)")
    axes[1].invert_yaxis()

    plt.suptitle("Form Scorer — Test Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "form_scorer_evaluation.png"
    plt.savefig(out, dpi=120); plt.close()
    logger.info(f"Saved: {out}")

    return {"form_mae": round(mae, 3), "form_r2": round(r2, 4)}


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results.update(evaluate_bilstm(
        Path(args.bilstm), Path(args.seqs), output_dir
    ))
    results.update(evaluate_form_scorer(
        Path(args.form), Path(args.release), output_dir
    ))

    out_json = output_dir / "evaluation_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✅ Evaluation complete → {output_dir}")
    logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bilstm",  default="models/bilstm/best_model.pt")
    parser.add_argument("--form",    default="models/form_scorer.pkl")
    parser.add_argument("--seqs",    default="data/processed/sequences")
    parser.add_argument("--release", default="data/processed/features/release_features.csv")
    parser.add_argument("--output",  default="docs/evaluation")
    args = parser.parse_args()
    main(args)

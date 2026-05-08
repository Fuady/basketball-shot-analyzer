"""
form_scorer.py
──────────────
Modeling Step 3: XGBoost form scorer and rule-based feedback engine.

Two components:
  1. FormScorer    — XGBoost predicts a 0–100 "form score" at the shot level
  2. FeedbackEngine — Rule-based system maps biomechanical readings to
                      specific corrective cues (e.g., "Elbow too wide at 48° — target ≤ 35°")

Usage:
    python src/modeling/form_scorer.py \
        --release data/processed/features/release_features.csv \
        --output  models/form_scorer.pkl
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── FORM SCORE FEATURES ──────────────────────────────────────────────────────
FORM_FEATURES = [
    "release_elbow_angle",
    "release_shoulder_angle",
    "release_knee_angle",
    "release_wrist_height",
    "release_trunk_lean",
    "release_shoulder_tilt",
    "release_wrist_speed",
    "pre_release_knee_bend",
    "pre_release_elbow_angle",
    "follow_through_elbow",
    "follow_through_wrist_drop",
    "max_wrist_speed",
    "max_elbow_extension",
    "min_knee_angle",
    "seq_length",
]


# ─── BIOMECHANICAL RULES FOR FEEDBACK ────────────────────────────────────────
# Format: (feature, low_thresh, high_thresh, low_msg, high_msg, ideal_range_str)
FEEDBACK_RULES = [
    {
        "feature": "release_elbow_angle",
        "ideal_low": 80, "ideal_high": 105,
        "low_msg":  "Elbow too bent at release ({val:.0f}°). Extend arm more — target 80–105°.",
        "high_msg": "Elbow too wide at release ({val:.0f}°). Tuck elbow in — target 80–105°.",
        "weight": 3,
    },
    {
        "feature": "pre_release_knee_bend",
        "ideal_low": 120, "ideal_high": 155,
        "low_msg":  "Not enough knee bend ({val:.0f}°). Bend knees deeper before shooting — target 120–155°.",
        "high_msg": "Squatting too deep ({val:.0f}°). Moderate knee bend — target 120–155°.",
        "weight": 2,
    },
    {
        "feature": "release_wrist_height",
        "ideal_low": 0.8, "ideal_high": 2.5,
        "low_msg":  "Release point too low (height={val:.2f}). Release higher — aim above your head.",
        "high_msg": None,
        "weight": 3,
    },
    {
        "feature": "release_shoulder_tilt",
        "ideal_low": 0.0, "ideal_high": 8.0,
        "low_msg":  None,
        "high_msg": "Shoulders are uneven at release ({val:.1f}° tilt). Keep shoulders level.",
        "weight": 2,
    },
    {
        "feature": "release_trunk_lean",
        "ideal_low": 0.0, "ideal_high": 15.0,
        "low_msg":  None,
        "high_msg": "Leaning back too far ({val:.1f}°). Stay balanced — lean slightly forward.",
        "weight": 2,
    },
    {
        "feature": "follow_through_elbow",
        "ideal_low": 150, "ideal_high": 180,
        "low_msg":  "Incomplete follow-through — elbow not fully extended ({val:.0f}°). Snap wrist through.",
        "high_msg": None,
        "weight": 2,
    },
    {
        "feature": "release_wrist_speed",
        "ideal_low": 1.5, "ideal_high": 99,
        "low_msg":  "Wrist snap too slow ({val:.2f}). Accelerate through the release — flick your wrist.",
        "high_msg": None,
        "weight": 2,
    },
]


class FeedbackEngine:
    """Rule-based biomechanical feedback generator."""

    def __init__(self, rules: list[dict] = None):
        self.rules = rules or FEEDBACK_RULES

    def generate(self, release_features: dict) -> list[dict]:
        """
        Generate corrective feedback items from release features.

        Returns list of dicts:
          {feature, value, message, severity (high/medium/low), weight}
        """
        feedback = []
        for rule in self.rules:
            feat = rule["feature"]
            val = release_features.get(feat)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue

            msg = None
            if val < rule["ideal_low"] and rule.get("low_msg"):
                msg = rule["low_msg"].format(val=val)
                severity = "high" if rule["weight"] >= 3 else "medium"
            elif val > rule["ideal_high"] and rule.get("high_msg"):
                msg = rule["high_msg"].format(val=val)
                severity = "high" if rule["weight"] >= 3 else "medium"

            if msg:
                feedback.append({
                    "feature": feat,
                    "value": round(float(val), 2),
                    "message": msg,
                    "severity": severity,
                    "weight": rule["weight"],
                })

        # Sort by severity + weight
        severity_order = {"high": 0, "medium": 1, "low": 2}
        feedback.sort(key=lambda x: (severity_order[x["severity"]], -x["weight"]))
        return feedback

    def score_form(self, release_features: dict) -> float:
        """
        Compute a simple rule-based form score (0–100) based on deviations.
        Each rule contributes penalty points. Perfect = 100.
        """
        total_weight = sum(r["weight"] for r in self.rules)
        penalty = 0.0

        for rule in self.rules:
            feat = rule["feature"]
            val = release_features.get(feat)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue

            ideal_low = rule["ideal_low"]
            ideal_high = rule["ideal_high"]
            weight = rule["weight"]

            if val < ideal_low:
                deviation = (ideal_low - val) / (ideal_low + 1e-9)
                penalty += weight * min(deviation, 1.0)
            elif val > ideal_high:
                deviation = (val - ideal_high) / (ideal_high + 1e-9)
                penalty += weight * min(deviation, 1.0)

        max_penalty = total_weight
        score = 100.0 * (1.0 - penalty / max_penalty)
        return max(0.0, min(100.0, score))


# ─── ML FORM SCORER ───────────────────────────────────────────────────────────

def create_form_score_labels(release_df: pd.DataFrame) -> pd.Series:
    """
    Create continuous form score labels (0–100) from rule-based engine.
    Used as regression targets for XGBoost.
    """
    engine = FeedbackEngine()
    scores = []
    for _, row in release_df.iterrows():
        score = engine.score_form(row.to_dict())
        scores.append(score)
    return pd.Series(scores, index=release_df.index)


def train_form_scorer(
    release_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[Pipeline, dict]:
    """Train XGBoost form scorer with cross-validation."""
    available = [f for f in FORM_FEATURES if f in release_df.columns]
    X = release_df[available].fillna(0)
    y = create_form_score_labels(release_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # Cross-validation
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(pipeline, X_train, y_train, cv=cv,
                               scoring="neg_mean_absolute_error", n_jobs=-1)
    logger.info(f"CV MAE: {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "test_mae": round(mean_absolute_error(y_test, y_pred), 3),
        "test_r2":  round(r2_score(y_test, y_pred), 3),
        "cv_mae_mean": round(cv_mae.mean(), 3),
        "cv_mae_std":  round(cv_mae.std(), 3),
    }

    logger.info("FORM SCORER METRICS")
    for k, v in metrics.items():
        logger.info(f"  {k:20s}: {v}")

    return pipeline, metrics


def main(args: argparse.Namespace) -> None:
    release_path = Path(args.release)
    output_path  = Path(args.output)

    if not release_path.exists():
        logger.error(f"Release features not found: {release_path}")
        sys.exit(1)

    release_df = pd.read_csv(release_path)

    mlflow.set_experiment("bball-form-scoring")
    with mlflow.start_run(run_name="xgb_form_scorer"):
        pipeline, metrics = train_form_scorer(release_df, args)
        mlflow.log_metrics(metrics)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "pipeline": pipeline,
            "feature_cols": [f for f in FORM_FEATURES if f in release_df.columns],
            "feedback_engine": FeedbackEngine(),
        }
        with open(output_path, "wb") as f:
            pickle.dump(artifact, f)

        mlflow.log_artifact(str(output_path), artifact_path="model")
        logger.info(f"\n✅ Form scorer saved: {output_path}")

    logger.info("Next: python src/modeling/evaluate.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="data/processed/features/release_features.csv")
    parser.add_argument("--output",  default="models/form_scorer.pkl")
    args = parser.parse_args()
    main(args)

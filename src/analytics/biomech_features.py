"""
biomech_features.py
───────────────────
Analytics: Engineer biomechanical features from normalized pose keypoints.

Features computed:
  - Joint angles  : elbow, knee, hip, wrist (using 3-point angle formula)
  - Angular velocity : frame-to-frame angle change
  - Alignment scores : shoulder tilt, hip-shoulder alignment
  - Temporal features: smoothed angle at key shot phases

Usage:
    python src/analytics/biomech_features.py \
        --keypoints data/processed/keypoints \
        --labels data/annotations/shot_labels.csv \
        --output data/processed/features
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── GEOMETRY ─────────────────────────────────────────────────────────────────

def angle_between_three_points(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> float:
    """
    Compute angle at point B formed by segments B→A and B→C.
    All points are (x, y) or (x, y, z) arrays.
    Returns angle in degrees.
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def get_point(row: pd.Series, name: str, dims: str = "xy") -> np.ndarray:
    """Extract landmark coordinates as numpy array."""
    coords = []
    for d in dims:
        val = row.get(f"{name}_{d}", np.nan)
        coords.append(float(val))
    return np.array(coords)


# ─── JOINT ANGLE COMPUTATION ──────────────────────────────────────────────────

def compute_joint_angles(row: pd.Series) -> dict[str, float]:
    """Compute all relevant joint angles for a single frame."""
    angles = {}

    # ── Shooting arm angles ──────────────────────────────────────────────────
    # Elbow angle: shoulder → elbow → wrist
    sh = get_point(row, "right_shoulder")
    el = get_point(row, "right_elbow")
    wr = get_point(row, "right_wrist")
    if not any(np.isnan(v) for v in [*sh, *el, *wr]):
        angles["elbow_angle"] = angle_between_three_points(sh, el, wr)

    # Shoulder angle: hip → shoulder → elbow
    hip = get_point(row, "right_hip")
    if not any(np.isnan(v) for v in [*hip, *sh, *el]):
        angles["shoulder_angle"] = angle_between_three_points(hip, sh, el)

    # ── Leg / drive angles ───────────────────────────────────────────────────
    # Right knee: hip → knee → ankle
    kn = get_point(row, "right_knee")
    an = get_point(row, "right_ankle")
    if not any(np.isnan(v) for v in [*hip, *kn, *an]):
        angles["right_knee_angle"] = angle_between_three_points(hip, kn, an)

    # Left knee
    lhip = get_point(row, "left_hip")
    lkn = get_point(row, "left_knee")
    lan = get_point(row, "left_ankle")
    if not any(np.isnan(v) for v in [*lhip, *lkn, *lan]):
        angles["left_knee_angle"] = angle_between_three_points(lhip, lkn, lan)

    # ── Trunk alignment ───────────────────────────────────────────────────────
    # Hip-shoulder trunk angle (forward lean)
    lsh = get_point(row, "left_shoulder")
    mid_hip = np.array([
        (row.get("left_hip_x", np.nan) + row.get("right_hip_x", np.nan)) / 2,
        (row.get("left_hip_y", np.nan) + row.get("right_hip_y", np.nan)) / 2,
    ])
    mid_sh = np.array([
        (row.get("left_shoulder_x", np.nan) + sh[0]) / 2,
        (row.get("left_shoulder_y", np.nan) + sh[1]) / 2,
    ])
    if not np.isnan(mid_hip).any() and not np.isnan(mid_sh).any():
        vec = mid_sh - mid_hip
        vertical = np.array([0, -1])  # pointing up
        cos_a = np.dot(vec, vertical) / (np.linalg.norm(vec) + 1e-9)
        angles["trunk_lean"] = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    # Shoulder tilt (horizontal alignment)
    if not any(np.isnan(v) for v in [*sh, *lsh]):
        delta_y = abs(sh[1] - lsh[1])
        delta_x = abs(sh[0] - lsh[0]) + 1e-9
        angles["shoulder_tilt"] = float(np.degrees(np.arctan(delta_y / delta_x)))

    # ── Wrist height (normalized by torso) ───────────────────────────────────
    wr_y = row.get("right_wrist_y", np.nan)
    angles["wrist_height_norm"] = float(wr_y) if not np.isnan(wr_y) else np.nan

    return angles


# ─── ANGULAR VELOCITY ─────────────────────────────────────────────────────────

def compute_angular_velocities(angles_df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    """
    Compute angular velocity (degrees per second) for each angle column.
    Adds new columns: {angle}_vel
    """
    angle_cols = [c for c in angles_df.columns
                  if not c.endswith("_vel") and c not in ["frame", "timestamp_ms"]]
    for col in angle_cols:
        angles_df[f"{col}_vel"] = (
            angles_df[col].diff() * fps
        ).fillna(0).clip(-3000, 3000)
    return angles_df


# ─── WRIST VELOCITY (release proxy) ───────────────────────────────────────────

def compute_wrist_speed(df: pd.DataFrame, fps: float = 30.0) -> pd.Series:
    """Compute wrist speed (pixels/sec, normalized units) as release proxy."""
    dx = df["right_wrist_x"].diff() * fps
    dy = df["right_wrist_y"].diff() * fps
    return np.sqrt(dx**2 + dy**2).fillna(0)


# ─── SMOOTHING ────────────────────────────────────────────────────────────────

def smooth_angles(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Apply Savitzky-Golay smoothing to angle columns."""
    angle_cols = [c for c in df.columns
                  if c not in ["frame", "timestamp_ms"] and not c.endswith("_vis")]
    for col in angle_cols:
        vals = df[col].values
        if len(vals) >= window and not np.all(np.isnan(vals)):
            w = window if window % 2 == 1 else window + 1
            w = min(w, len(vals) if len(vals) % 2 == 1 else len(vals) - 1)
            if w >= 3:
                try:
                    df[col] = savgol_filter(
                        np.nan_to_num(vals, nan=np.nanmean(vals)),
                        window_length=w, polyorder=2
                    )
                except Exception:
                    pass
    return df


# ─── FULL PIPELINE ────────────────────────────────────────────────────────────

def process_keypoint_csv(
    csv_path: Path,
    shot_type: str,
    outcome: int,
    fps: float = 30.0,
) -> pd.DataFrame:
    """
    Compute all biomechanical features for a single normalized keypoint CSV.
    Returns a DataFrame with one row per frame + joint angle features.
    """
    df = pd.read_csv(csv_path)

    # Compute joint angles per frame
    angle_rows = []
    for _, row in df.iterrows():
        angles = compute_joint_angles(row)
        angles["frame"] = int(row["frame"])
        angles["timestamp_ms"] = float(row.get("timestamp_ms", 0))
        angle_rows.append(angles)

    angles_df = pd.DataFrame(angle_rows).sort_values("frame").reset_index(drop=True)

    # Smooth angles
    angles_df = smooth_angles(angles_df)

    # Angular velocities
    angles_df = compute_angular_velocities(angles_df, fps)

    # Wrist speed
    angles_df["wrist_speed"] = compute_wrist_speed(df, fps).values[:len(angles_df)]

    # Metadata
    angles_df["shot_type"] = shot_type
    angles_df["outcome"] = outcome
    angles_df["source"] = csv_path.stem

    return angles_df


def main(args: argparse.Namespace) -> None:
    keypoints_dir = Path(args.keypoints)
    labels_path = Path(args.labels)
    output_dir = Path(args.output)

    if not keypoints_dir.exists():
        logger.error(f"Keypoints dir not found: {keypoints_dir}")
        sys.exit(1)
    if not labels_path.exists():
        logger.error(f"Labels not found: {labels_path}")
        sys.exit(1)

    labels_df = pd.read_csv(labels_path)
    label_map = dict(zip(labels_df["video_stem"], labels_df["outcome"]))
    shot_type_map = dict(zip(labels_df["video_stem"], labels_df["shot_type"]))

    csv_paths = list(keypoints_dir.rglob("*_norm.csv"))
    logger.info(f"Processing {len(csv_paths)} keypoint CSVs...")

    all_dfs = []
    for csv_path in tqdm(csv_paths, desc="Features"):
        stem = csv_path.stem.replace("_norm", "")
        outcome = label_map.get(stem)
        if outcome is None:
            logger.debug(f"No label for: {stem}")
            continue
        shot_type = shot_type_map.get(stem, "unknown")
        try:
            feat_df = process_keypoint_csv(csv_path, shot_type, int(outcome))
            feat_df["video_stem"] = stem
            all_dfs.append(feat_df)
        except Exception as e:
            logger.warning(f"Failed to process {csv_path.name}: {e}")

    if not all_dfs:
        logger.error("No features computed.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "biomech_features.csv"
    combined.to_csv(out_path, index=False)

    logger.info(f"\n✅ Features saved: {out_path}")
    logger.info(f"   Total rows  : {len(combined):,}")
    logger.info(f"   Total videos: {combined['video_stem'].nunique()}")
    logger.info(f"   Feature cols: {len(combined.columns)}")
    logger.info("\nNext: python src/analytics/release_detector.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engineer biomechanical features from keypoints")
    parser.add_argument("--keypoints", type=str, default="data/processed/keypoints")
    parser.add_argument("--labels", type=str, default="data/annotations/shot_labels.csv")
    parser.add_argument("--output", type=str, default="data/processed/features")
    args = parser.parse_args()
    main(args)

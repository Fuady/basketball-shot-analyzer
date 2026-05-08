"""
normalize_poses.py
──────────────────
Data Engineering Step 3: Normalize raw MediaPipe keypoints.

Normalization strategy:
  1. Hip-centered: subtract mid-hip position so player is always at origin
  2. Scale-invariant: divide by torso length (mid-hip to mid-shoulder distance)
  3. Handle missing frames: forward-fill then drop leading NaN frames
  4. Mirror left-handed players to right-handed convention

This makes features invariant to:
  - Player position in frame
  - Camera distance / zoom level
  - Player height

Usage:
    python src/data_engineering/normalize_poses.py \
        --input data/raw/poses \
        --output data/processed/keypoints
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Key landmark indices for normalization reference points
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_WRIST_IDX = 15
RIGHT_WRIST_IDX = 16

# All coordinate columns (x, y, z) — exclude visibility
def get_coord_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]


def detect_handedness(df: pd.DataFrame) -> str:
    """
    Heuristic: shooting hand is the one with higher mean x position
    for right-side shots (most players are right-handed).
    Use wrist x-positions to determine dominant shooting hand.
    """
    rw = df["right_wrist_x"].dropna()
    lw = df["left_wrist_x"].dropna()
    if rw.empty or lw.empty:
        return "right"
    # If right wrist is consistently higher in frame → right-handed
    return "right" if rw.mean() > lw.mean() else "left"


def mirror_to_right_handed(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror all x coordinates for left-handed players."""
    df = df.copy()
    x_cols = [c for c in df.columns if c.endswith("_x")]
    df[x_cols] = 1.0 - df[x_cols]
    # Swap left/right landmarks
    swap_pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
        ("left_heel", "right_heel"),
        ("left_foot_index", "right_foot_index"),
    ]
    for left, right in swap_pairs:
        for coord in ["x", "y", "z", "vis"]:
            lc, rc = f"{left}_{coord}", f"{right}_{coord}"
            if lc in df.columns and rc in df.columns:
                df[lc], df[rc] = df[rc].copy(), df[lc].copy()
    return df


def normalize_frame(row: pd.Series) -> pd.Series:
    """
    Normalize a single frame's keypoints:
      1. Center on mid-hip
      2. Scale by torso length
    """
    row = row.copy()

    # Mid-hip reference point
    hip_x = (row.get("left_hip_x", np.nan) + row.get("right_hip_x", np.nan)) / 2
    hip_y = (row.get("left_hip_y", np.nan) + row.get("right_hip_y", np.nan)) / 2
    hip_z = (row.get("left_hip_z", np.nan) + row.get("right_hip_z", np.nan)) / 2

    # Mid-shoulder reference point
    sh_x = (row.get("left_shoulder_x", np.nan) + row.get("right_shoulder_x", np.nan)) / 2
    sh_y = (row.get("left_shoulder_y", np.nan) + row.get("right_shoulder_y", np.nan)) / 2
    sh_z = (row.get("left_shoulder_z", np.nan) + row.get("right_shoulder_z", np.nan)) / 2

    # Torso length (scale reference)
    torso_length = np.sqrt(
        (sh_x - hip_x) ** 2 + (sh_y - hip_y) ** 2 + (sh_z - hip_z) ** 2
    )
    if np.isnan(torso_length) or torso_length < 1e-6:
        return row  # can't normalize this frame

    # Apply centering and scaling to all coordinate columns
    x_cols = [c for c in row.index if c.endswith("_x")]
    y_cols = [c for c in row.index if c.endswith("_y")]
    z_cols = [c for c in row.index if c.endswith("_z")]

    row[x_cols] = (row[x_cols] - hip_x) / torso_length
    row[y_cols] = (row[y_cols] - hip_y) / torso_length
    row[z_cols] = (row[z_cols] - hip_z) / torso_length

    return row


def normalize_pose_csv(
    csv_path: Path,
    output_path: Path,
    mirror_left_handed: bool = True,
    fill_method: str = "ffill",
) -> dict:
    """
    Normalize a single pose CSV file.
    Returns metadata dict.
    """
    df = pd.read_csv(csv_path)
    original_len = len(df)

    # Forward-fill missing pose frames
    coord_cols = get_coord_cols(df)
    df[coord_cols] = df[coord_cols].ffill().bfill()

    # Drop rows still NaN after fill (leading missing frames)
    df = df.dropna(subset=["nose_x"])
    after_fill_len = len(df)

    # Mirror left-handed players
    mirrored = False
    if mirror_left_handed:
        handedness = detect_handedness(df)
        if handedness == "left":
            df = mirror_to_right_handed(df)
            mirrored = True

    # Normalize each frame
    feature_cols = [c for c in df.columns if c not in ["frame", "timestamp_ms"]]
    norm_rows = []
    for _, row in df.iterrows():
        norm_rows.append(normalize_frame(row))
    df_norm = pd.DataFrame(norm_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_norm.to_csv(output_path, index=False)

    return {
        "source": str(csv_path),
        "output": str(output_path),
        "original_frames": original_len,
        "normalized_frames": after_fill_len,
        "mirrored": mirrored,
        "drop_rate": round(1 - after_fill_len / max(original_len, 1), 3),
    }


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    pose_csvs = list(input_dir.rglob("*_poses.csv"))
    if not pose_csvs:
        logger.error(f"No pose CSVs found in {input_dir}")
        sys.exit(1)

    logger.info(f"Normalizing {len(pose_csvs)} pose CSVs...")
    results = []

    for csv_path in tqdm(pose_csvs, desc="Normalizing"):
        rel = csv_path.relative_to(input_dir)
        out_path = output_dir / rel.parent / (csv_path.stem.replace("_poses", "_norm") + ".csv")
        meta = normalize_pose_csv(csv_path, out_path)
        results.append(meta)

    avg_drop = sum(r["drop_rate"] for r in results) / max(len(results), 1)
    mirrored = sum(1 for r in results if r["mirrored"])
    logger.info(f"\n✅ Normalized {len(results)} files")
    logger.info(f"   Avg frame drop rate : {avg_drop:.1%}")
    logger.info(f"   Left-hand mirrored  : {mirrored}")
    logger.info("Next: python src/data_engineering/label_shots.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize MediaPipe pose keypoints")
    parser.add_argument("--input", type=str, default="data/raw/poses")
    parser.add_argument("--output", type=str, default="data/processed/keypoints")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Disable left-hand mirroring")
    args = parser.parse_args()
    args.mirror_left_handed = not args.no_mirror
    main(args)

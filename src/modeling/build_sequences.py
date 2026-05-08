"""
build_sequences.py
──────────────────
Modeling Step 1: Build fixed-length sequences for LSTM training.

For each shot:
  - Takes the T frames centered around the release frame
  - Pads shorter sequences with zeros
  - Saves as numpy arrays: X_train.npy, y_train.npy, etc.

Usage:
    python src/modeling/build_sequences.py \
        --features data/processed/features/biomech_features.csv \
        --release  data/processed/features/release_features.csv \
        --output   data/processed/sequences \
        --seq-len  30
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Features used as LSTM input at each timestep
SEQUENCE_FEATURES = [
    "elbow_angle",
    "shoulder_angle",
    "right_knee_angle",
    "left_knee_angle",
    "trunk_lean",
    "shoulder_tilt",
    "wrist_height_norm",
    "wrist_speed",
    "elbow_angle_vel",
    "shoulder_angle_vel",
    "right_knee_angle_vel",
    "wrist_height_norm_vel",
]


def extract_sequence(
    group: pd.DataFrame,
    release_frame: int,
    seq_len: int,
    features: list[str],
) -> np.ndarray:
    """
    Extract a fixed-length sequence centered on the release frame.

    Strategy:
      - Take seq_len frames ending at release_frame
      - Pad from start with zeros if shot is shorter
    Returns: (seq_len, n_features) array
    """
    group = group.sort_values("frame").reset_index(drop=True)
    available_feats = [f for f in features if f in group.columns]

    # Find index of release frame
    try:
        rel_idx = group.index[group["frame"] == release_frame][0]
    except IndexError:
        rel_idx = int(len(group) * 0.6)

    # Extract window ending at release frame
    start_idx = max(0, rel_idx - seq_len + 1)
    window = group.iloc[start_idx:rel_idx + 1][available_feats].values.astype(np.float32)

    # Fill NaN with column mean (or 0)
    col_means = np.nanmean(window, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    for col_i in range(window.shape[1]):
        nan_mask = np.isnan(window[:, col_i])
        window[nan_mask, col_i] = col_means[col_i]

    # Pad to seq_len with zeros at the start
    if len(window) < seq_len:
        pad = np.zeros((seq_len - len(window), len(available_feats)), dtype=np.float32)
        window = np.vstack([pad, window])

    return window[:seq_len]


def build_sequences(
    features_df: pd.DataFrame,
    release_df: pd.DataFrame,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build X, y arrays for all shots.

    Returns:
        X : (n_shots, seq_len, n_features)
        y : (n_shots,) binary outcome
        video_stems: list of video IDs
    """
    release_map = dict(zip(release_df["video_stem"], release_df["release_frame"]))

    X_list, y_list, stems = [], [], []
    skipped = 0

    for video_stem, group in features_df.groupby("video_stem"):
        release_frame = release_map.get(video_stem)
        if release_frame is None:
            skipped += 1
            continue

        outcome = int(group["outcome"].iloc[0])
        seq = extract_sequence(group, int(release_frame), seq_len, SEQUENCE_FEATURES)

        X_list.append(seq)
        y_list.append(outcome)
        stems.append(video_stem)

    if not X_list:
        logger.error("No sequences built. Check that features and release files are aligned.")
        sys.exit(1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info(f"Built {len(X)} sequences (skipped {skipped} without release labels)")
    logger.info(f"  X shape : {X.shape}  (shots × timesteps × features)")
    logger.info(f"  y dist  : Makes={int(y.sum())}, Misses={int((1-y).sum())}")
    return X, y, stems


def normalize_sequences(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit StandardScaler on training set, apply to all splits.
    Operates on flattened features axis (not time axis).
    """
    n_train, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_2d)

    X_train = scaler.transform(X_train_2d).reshape(n_train, seq_len, n_feat)
    X_val   = scaler.transform(X_val.reshape(-1, n_feat)).reshape(len(X_val), seq_len, n_feat)
    X_test  = scaler.transform(X_test.reshape(-1, n_feat)).reshape(len(X_test), seq_len, n_feat)

    scaler_path = output_dir / "sequence_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved: {scaler_path}")
    return X_train, X_val, X_test


def main(args: argparse.Namespace) -> None:
    features_path = Path(args.features)
    release_path  = Path(args.release)
    output_dir    = Path(args.output)

    for p in [features_path, release_path]:
        if not p.exists():
            logger.error(f"Not found: {p}")
            sys.exit(1)

    features_df = pd.read_csv(features_path)
    release_df  = pd.read_csv(release_path)

    X, y, stems = build_sequences(features_df, release_df, args.seq_len)

    # Stratified split: 70/15/15
    X_tv, X_test, y_tv, y_test, s_tv, s_test = train_test_split(
        X, y, stems, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
        X_tv, y_tv, s_tv, test_size=0.176, stratify=y_tv, random_state=42  # 0.176 ≈ 15/85
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize
    X_train, X_val, X_test = normalize_sequences(X_train, X_val, X_test, output_dir)

    # Save arrays
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy",   X_val)
    np.save(output_dir / "X_test.npy",  X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_val.npy",   y_val)
    np.save(output_dir / "y_test.npy",  y_test)

    # Save feature names
    feat_path = output_dir / "feature_names.txt"
    used_feats = [f for f in SEQUENCE_FEATURES if f in features_df.columns]
    feat_path.write_text("\n".join(used_feats))

    logger.info(f"\n✅ Sequences saved → {output_dir}")
    logger.info(f"   Train : {X_train.shape} | Makes: {int(y_train.sum())}")
    logger.info(f"   Val   : {X_val.shape}   | Makes: {int(y_val.sum())}")
    logger.info(f"   Test  : {X_test.shape}  | Makes: {int(y_test.sum())}")
    logger.info(f"   Features ({len(used_feats)}): {used_feats}")
    logger.info("\nNext: python src/modeling/train_bilstm.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build LSTM sequences from biomech features")
    parser.add_argument("--features", default="data/processed/features/biomech_features.csv")
    parser.add_argument("--release",  default="data/processed/features/release_features.csv")
    parser.add_argument("--output",   default="data/processed/sequences")
    parser.add_argument("--seq-len",  type=int, default=30,
                        help="Sequence length in frames (default: 30 = 1 second at 30fps)")
    args = parser.parse_args()
    main(args)

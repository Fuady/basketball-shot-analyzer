"""
release_detector.py
───────────────────
Analytics: Automatically detect the ball release frame from pose data.

The release frame is the moment when the wrist reaches peak upward velocity
and begins to snap forward — the critical instant for shot mechanics analysis.

Detection algorithm:
  1. Compute frame-by-frame wrist velocity (x, y components)
  2. Smooth with Savitzky-Golay filter
  3. Find frame with peak upward wrist velocity (negative y in image coords)
  4. Confirm with elbow extension pattern (elbow angle increasing toward release)
  5. Fallback: use top-quartile wrist speed frame if above fails

Usage:
    python src/analytics/release_detector.py \
        --features data/processed/features/biomech_features.csv \
        --keypoints data/processed/keypoints \
        --output data/processed/features/release_frames.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Window around release frame to extract for the LSTM (frames before release)
PRE_RELEASE_WINDOW = 30   # 1 second at 30fps
POST_RELEASE_WINDOW = 5


def detect_release_frame(
    keypoint_df: pd.DataFrame,
    fps: float = 30.0,
    smooth_window: int = 7,
) -> dict:
    """
    Detect release frame for a single shot.

    Parameters
    ----------
    keypoint_df : DataFrame with normalized keypoints (one row per frame)
    fps         : Video frame rate
    smooth_window : Savitzky-Golay window for smoothing wrist velocity

    Returns
    -------
    dict with release_frame, release_ts_ms, method, confidence
    """
    if len(keypoint_df) < 10:
        return {"release_frame": len(keypoint_df) // 2, "method": "fallback_short",
                "confidence": 0.1}

    wrist_y = keypoint_df["right_wrist_y"].ffill().values
    wrist_x = keypoint_df["right_wrist_x"].ffill().values
    elbow_angle = keypoint_df.get("elbow_angle", pd.Series(np.nan, index=keypoint_df.index))

    # Wrist velocity (negative y = moving upward in image coords)
    wy_vel = np.diff(wrist_y, prepend=wrist_y[0]) * fps
    wx_vel = np.diff(wrist_x, prepend=wrist_x[0]) * fps
    wrist_speed = np.sqrt(wx_vel**2 + wy_vel**2)

    # Smooth
    if len(wrist_y) >= smooth_window:
        w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        wy_vel_smooth = savgol_filter(wy_vel, window_length=w, polyorder=2)
        speed_smooth = savgol_filter(wrist_speed, window_length=w, polyorder=2)
    else:
        wy_vel_smooth = wy_vel
        speed_smooth = wrist_speed

    # Strategy 1: Peak upward wrist velocity in the top half of the video
    # (release typically happens in the upper portion of the shot motion)
    search_start = len(keypoint_df) // 4
    search_end = int(len(keypoint_df) * 0.9)
    search_region = wy_vel_smooth[search_start:search_end]

    # Most negative y velocity = fastest upward wrist movement = release
    peaks, props = find_peaks(-search_region, height=0.05, distance=5)

    if len(peaks) > 0:
        # Choose peak with highest upward velocity
        best_peak = peaks[np.argmax(-search_region[peaks])]
        release_frame = best_peak + search_start
        confidence = min(1.0, abs(search_region[best_peak]) / (np.std(wy_vel_smooth) + 1e-6))
        method = "peak_wrist_velocity"
    else:
        # Strategy 2: Maximum wrist speed in search region
        release_frame = np.argmax(speed_smooth[search_start:search_end]) + search_start
        confidence = 0.5
        method = "max_wrist_speed"

    # Refine with elbow extension if available
    if not elbow_angle.isna().all():
        elbow_vals = elbow_angle.ffill().values
        # Look for frame where elbow is most extended (near max) in ±5 frame window
        lo = max(0, release_frame - 5)
        hi = min(len(elbow_vals) - 1, release_frame + 5)
        local_max_elbow = lo + np.argmax(elbow_vals[lo:hi + 1])
        if abs(elbow_vals[local_max_elbow] - elbow_vals[release_frame]) < 20:
            release_frame = int(local_max_elbow)
            method += "+elbow_refinement"
            confidence = min(1.0, confidence + 0.1)

    ts_ms = float(keypoint_df["timestamp_ms"].iloc[release_frame]) \
        if "timestamp_ms" in keypoint_df.columns else release_frame * (1000 / fps)

    return {
        "release_frame": int(release_frame),
        "release_ts_ms": round(ts_ms, 1),
        "method": method,
        "confidence": round(confidence, 3),
        "total_frames": len(keypoint_df),
        "pre_release_frames": min(release_frame, PRE_RELEASE_WINDOW),
    }


def extract_release_window(
    keypoint_df: pd.DataFrame,
    release_frame: int,
    pre_window: int = PRE_RELEASE_WINDOW,
    post_window: int = POST_RELEASE_WINDOW,
) -> pd.DataFrame:
    """
    Extract the shot window: pre_window frames before release + post_window after.
    Pad with zeros if the window extends beyond the video start.
    """
    start = max(0, release_frame - pre_window)
    end = min(len(keypoint_df), release_frame + post_window + 1)
    window = keypoint_df.iloc[start:end].copy()

    # Pad beginning if needed
    if release_frame < pre_window:
        pad_len = pre_window - release_frame
        pad_df = pd.DataFrame(
            np.zeros((pad_len, len(window.columns))),
            columns=window.columns,
        )
        window = pd.concat([pad_df, window], ignore_index=True)

    return window.reset_index(drop=True)


def process_all_shots(
    keypoints_dir: Path,
    features_path: Path,
) -> pd.DataFrame:
    """
    Detect release frames for all shots and return a summary DataFrame.
    """
    if features_path.exists():
        features_df = pd.read_csv(features_path)
        video_stems = features_df["video_stem"].unique() \
            if "video_stem" in features_df.columns else []
    else:
        video_stems = []

    csv_paths = list(keypoints_dir.rglob("*_norm.csv"))
    records = []

    for csv_path in tqdm(csv_paths, desc="Detecting release frames"):
        stem = csv_path.stem.replace("_norm", "")
        try:
            kp_df = pd.read_csv(csv_path)

            # Merge elbow angle from features if available
            if len(video_stems) > 0 and stem in video_stems:
                vid_features = features_df[features_df["video_stem"] == stem]
                if "elbow_angle" in vid_features.columns:
                    kp_df = kp_df.merge(
                        vid_features[["frame", "elbow_angle"]].drop_duplicates("frame"),
                        on="frame",
                        how="left",
                    )

            result = detect_release_frame(kp_df)
            result["video_stem"] = stem
            result["shot_type"] = csv_path.parent.name
            records.append(result)
        except Exception as e:
            logger.warning(f"Failed {csv_path.name}: {e}")

    return pd.DataFrame(records)


def main(args: argparse.Namespace) -> None:
    keypoints_dir = Path(args.keypoints)
    features_path = Path(args.features)
    output_path = Path(args.output)

    if not keypoints_dir.exists():
        logger.error(f"Keypoints dir not found: {keypoints_dir}")
        sys.exit(1)

    logger.info("Detecting release frames...")
    results_df = process_all_shots(keypoints_dir, features_path)

    if results_df.empty:
        logger.error("No release frames detected.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    avg_conf = results_df["confidence"].mean()
    method_counts = results_df["method"].value_counts().to_dict()

    logger.info(f"\n✅ Release frames detected: {len(results_df)}")
    logger.info(f"   Average confidence : {avg_conf:.3f}")
    logger.info(f"   Detection methods  : {method_counts}")
    logger.info(f"   Saved to           : {output_path}")
    logger.info("\nNext: python src/modeling/build_sequences.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect release frames in shot videos")
    parser.add_argument("--keypoints", type=str, default="data/processed/keypoints")
    parser.add_argument("--features", type=str,
                        default="data/processed/features/biomech_features.csv")
    parser.add_argument("--output", type=str,
                        default="data/processed/features/release_frames.csv")
    args = parser.parse_args()
    main(args)


# ─── ADDITIONAL HELPERS (used by inference.py and tests) ──────────────────────

def compute_release_features(group: pd.DataFrame, release_frame: int) -> dict:
    """
    Extract biomechanical snapshot at the release frame.
    This is imported by inference.py for per-shot analysis.
    """
    release_row = group[group["frame"] == release_frame]
    if release_row.empty:
        idx = (group["frame"] - release_frame).abs().idxmin()
        release_row = group.loc[[idx]]

    row = release_row.iloc[0]
    n = len(group)
    pos = group.index.get_loc(release_row.index[0])

    def _win_mean(col, s, e):
        start = max(0, pos + s); end = min(n, pos + e)
        if start >= end or col not in group.columns:
            return float("nan")
        return float(group[col].iloc[start:end].mean())

    return {
        "release_frame":           release_frame,
        "release_position_pct":    round(pos / max(n - 1, 1), 3),
        "release_elbow_angle":     row.get("elbow_angle",       float("nan")),
        "release_shoulder_angle":  row.get("shoulder_angle",    float("nan")),
        "release_knee_angle":      row.get("right_knee_angle",  float("nan")),
        "release_wrist_height":    row.get("wrist_height_norm", float("nan")),
        "release_trunk_lean":      row.get("trunk_lean",        float("nan")),
        "release_shoulder_tilt":   row.get("shoulder_tilt",     float("nan")),
        "release_wrist_speed":     row.get("wrist_speed",       float("nan")),
        "pre_release_knee_bend":   _win_mean("right_knee_angle", -15, -5),
        "pre_release_elbow_angle": _win_mean("elbow_angle",      -10, -3),
        "follow_through_elbow":    _win_mean("elbow_angle",        3, 10),
        "follow_through_wrist_drop": _win_mean("wrist_speed",     3,  8),
        "release_elbow_vel":       row.get("elbow_angle_vel",   float("nan")),
        "seq_length":              n,
        "max_wrist_speed":         float(group["wrist_speed"].max()) if "wrist_speed"      in group.columns else float("nan"),
        "max_elbow_extension":     float(group["elbow_angle"].max())  if "elbow_angle"     in group.columns else float("nan"),
        "min_knee_angle":          float(group["right_knee_angle"].min()) if "right_knee_angle" in group.columns else float("nan"),
    }


def arc_curvature_check(group: pd.DataFrame) -> float:
    """
    Simple arc curvature metric used in tests.
    Returns mean absolute second derivative of wrist height (proxy for arc shape).
    """
    if "wrist_height_norm" not in group.columns:
        return 0.0
    vals = group["wrist_height_norm"].ffill().fillna(0).values
    if len(vals) < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(vals, n=2))))

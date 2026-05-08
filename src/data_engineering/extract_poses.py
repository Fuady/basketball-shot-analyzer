"""
extract_poses.py
────────────────
Data Engineering Step 2: Extract 33 body keypoints per frame using MediaPipe Pose.

For each video:
  - Runs MediaPipe Pose on every frame
  - Saves landmark (x, y, z, visibility) × 33 points per frame to CSV
  - Also saves an annotated video with skeleton overlay (optional)

Output CSV columns:
  frame, timestamp_ms, [landmark_0_x, landmark_0_y, landmark_0_z, landmark_0_vis, ...] × 33

MediaPipe landmark indices:
  https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

Usage:
    python src/data_engineering/extract_poses.py \
        --input data/raw/videos \
        --output data/raw/poses \
        --save-video
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
N_LANDMARKS = 33

# MediaPipe landmark names (index → name)
LANDMARK_NAMES = [lm.name.lower() for lm in mp_pose.PoseLandmark]

# Build CSV header
POSE_CSV_HEADER = ["frame", "timestamp_ms"]
for i, name in enumerate(LANDMARK_NAMES):
    for coord in ["x", "y", "z", "vis"]:
        POSE_CSV_HEADER.append(f"{name}_{coord}")


def extract_poses_from_video(
    video_path: Path,
    output_dir: Path,
    save_annotated: bool = False,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[list[dict], int]:
    """
    Extract MediaPipe poses from all frames of a video.

    Returns:
        (list of frame records, total frames processed)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open: {video_path}")
        return [], 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up annotated video writer
    writer = None
    if save_annotated:
        ann_path = output_dir / (video_path.stem + "_annotated.mp4")
        writer = cv2.VideoWriter(
            str(ann_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

    records = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        with tqdm(total=total, desc=video_path.name[:50], unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                # MediaPipe expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = pose.process(rgb)
                rgb.flags.writeable = True

                record = {"frame": frame_idx, "timestamp_ms": round(timestamp_ms, 1)}

                if results.pose_landmarks:
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        name = LANDMARK_NAMES[i]
                        record[f"{name}_x"] = round(lm.x, 6)
                        record[f"{name}_y"] = round(lm.y, 6)
                        record[f"{name}_z"] = round(lm.z, 6)
                        record[f"{name}_vis"] = round(lm.visibility, 4)
                else:
                    # Fill NaN for frames where pose is not detected
                    for i, name in enumerate(LANDMARK_NAMES):
                        for coord in ["x", "y", "z", "vis"]:
                            record[f"{name}_{coord}"] = float("nan")

                records.append(record)

                # Draw skeleton on annotated video
                if writer and results.pose_landmarks:
                    annotated = frame.copy()
                    mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                    cv2.putText(
                        annotated,
                        f"Frame: {frame_idx}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    )
                    writer.write(annotated)

                frame_idx += 1
                pbar.update(1)

    cap.release()
    if writer:
        writer.release()

    detected = sum(1 for r in records if not np.isnan(r.get("nose_x", float("nan"))))
    logger.info(f"  {video_path.name}: {detected}/{frame_idx} frames with pose detected")
    return records, frame_idx


def save_pose_csv(records: list[dict], output_path: Path) -> None:
    """Save pose records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return

    fieldnames = POSE_CSV_HEADER
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def process_directory(
    input_dir: Path,
    output_dir: Path,
    save_annotated: bool,
) -> list[dict]:
    """Process all videos in input_dir, preserving subdir structure."""
    video_paths = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    ]

    if not video_paths:
        logger.error(f"No videos found in {input_dir}")
        return []

    logger.info(f"Found {len(video_paths)} videos")
    all_meta = []

    for vp in video_paths:
        rel = vp.relative_to(input_dir)
        shot_type = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        out_dir = output_dir / shot_type
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / (vp.stem + "_poses.csv")

        if csv_path.exists():
            logger.info(f"  Skipping (already exists): {csv_path.name}")
            continue

        records, total_frames = extract_poses_from_video(
            video_path=vp,
            output_dir=out_dir,
            save_annotated=save_annotated,
        )

        if records:
            save_pose_csv(records, csv_path)
            all_meta.append({
                "video": vp.name,
                "shot_type": shot_type,
                "total_frames": total_frames,
                "pose_csv": str(csv_path),
                "detection_rate": sum(
                    1 for r in records if not np.isnan(r.get("nose_x", float("nan")))
                ) / max(total_frames, 1),
            })

    return all_meta


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    logger.info(f"Extracting poses from: {input_dir}")
    meta = process_directory(input_dir, output_dir, args.save_video)

    import json
    manifest_path = output_dir / "poses_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(meta, f, indent=2)

    avg_det = sum(m["detection_rate"] for m in meta) / max(len(meta), 1)
    logger.info(f"\n✅ Processed {len(meta)} videos | Avg detection rate: {avg_det:.2%}")
    logger.info("Next: python src/data_engineering/normalize_poses.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MediaPipe poses from basketball videos")
    parser.add_argument("--input", type=str, default="data/raw/videos")
    parser.add_argument("--output", type=str, default="data/raw/poses")
    parser.add_argument("--save-video", action="store_true",
                        help="Save annotated skeleton overlay video")
    parser.add_argument("--min-detection", type=float, default=0.5)
    parser.add_argument("--min-tracking", type=float, default=0.5)
    args = parser.parse_args()
    main(args)

"""
label_shots.py
──────────────
Data Engineering Step 4: Label shot outcomes (make / miss).

Three labeling methods:
  1. --auto    : Parse title/filename for keywords ("made", "miss", "good", etc.)
  2. --manual  : Interactive frame-by-frame labeler using OpenCV
  3. --synthetic: Generate synthetic labels for pipeline testing

Output: data/annotations/shot_labels.csv
  columns: video_stem, shot_type, outcome (make=1, miss=0), label_source

Usage:
    python src/data_engineering/label_shots.py --auto --input data/raw/videos
    python src/data_engineering/label_shots.py --manual --input data/raw/videos
    python src/data_engineering/label_shots.py --synthetic --keypoints data/processed/keypoints
"""

import argparse
import csv
import logging
import random
import re
import sys
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAKE_KEYWORDS = {"make", "made", "swish", "good", "score", "bucket", "in", "splash"}
MISS_KEYWORDS = {"miss", "missed", "brick", "airball", "clank", "off", "rim", "short"}

SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_outcome_from_title(title: str) -> int | None:
    """Return 1 (make), 0 (miss), or None (unknown) from video title."""
    title_lower = title.lower()
    words = set(re.findall(r'\w+', title_lower))
    make_hits = words & MAKE_KEYWORDS
    miss_hits = words & MISS_KEYWORDS
    if make_hits and not miss_hits:
        return 1
    if miss_hits and not make_hits:
        return 0
    return None


def auto_label(input_dir: Path) -> list[dict]:
    """Auto-label videos from filename/title keywords."""
    records = []
    for vp in sorted(input_dir.rglob("*")):
        if vp.suffix.lower() not in SUPPORTED_EXTS:
            continue
        shot_type = vp.parent.name
        outcome = parse_outcome_from_title(vp.stem)
        if outcome is not None:
            records.append({
                "video_stem": vp.stem,
                "video_path": str(vp),
                "shot_type": shot_type,
                "outcome": outcome,
                "label_source": "auto_title",
            })
            symbol = "✅" if outcome == 1 else "❌"
            logger.info(f"  {symbol} {vp.name[:60]}")
        else:
            logger.debug(f"  ❓ Ambiguous: {vp.name}")

    logger.info(f"\nAuto-labeled: {len(records)} videos")
    makes = sum(1 for r in records if r["outcome"] == 1)
    misses = sum(1 for r in records if r["outcome"] == 0)
    logger.info(f"  Makes: {makes} | Misses: {misses}")
    return records


def manual_label(input_dir: Path) -> list[dict]:
    """
    Interactive labeler: shows video frames, user presses:
      M = make  |  X = miss  |  S = skip  |  Q = quit
    """
    records = []
    video_paths = [
        vp for vp in sorted(input_dir.rglob("*"))
        if vp.suffix.lower() in SUPPORTED_EXTS
    ]

    logger.info(f"Manual labeling {len(video_paths)} videos")
    logger.info("Controls: [M]=Make  [X]=Miss  [S]=Skip  [Q]=Quit")

    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Show frame near the end (likely shows shot result)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, int(total * 0.8)))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            continue

        # Display frame with instructions
        display = cv2.resize(frame, (800, 450))
        cv2.putText(display, f"{vp.name[:50]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, "[M]=Make  [X]=Miss  [S]=Skip  [Q]=Quit",
                    (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Basketball Shot Labeler", display)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            logger.info("Labeling stopped by user.")
            break
        elif key == ord("m"):
            outcome = 1
            logger.info(f"  ✅ Make: {vp.name}")
        elif key == ord("x"):
            outcome = 0
            logger.info(f"  ❌ Miss: {vp.name}")
        else:
            logger.info(f"  ⏭ Skip: {vp.name}")
            continue

        records.append({
            "video_stem": vp.stem,
            "video_path": str(vp),
            "shot_type": vp.parent.name,
            "outcome": outcome,
            "label_source": "manual",
        })

    cv2.destroyAllWindows()
    return records


def synthetic_labels(keypoints_dir: Path) -> list[dict]:
    """
    Generate synthetic make/miss labels for pipeline testing.
    Assigns labels based on a simple heuristic (random with 55% make rate,
    matching typical NBA free throw percentage).
    """
    logger.warning("Generating SYNTHETIC labels — for pipeline testing only!")
    records = []
    csvs = list(keypoints_dir.rglob("*_norm.csv"))
    if not csvs:
        logger.error(f"No normalized keypoint CSVs found in {keypoints_dir}")
        return []

    for csv_path in csvs:
        outcome = 1 if random.random() < 0.55 else 0
        records.append({
            "video_stem": csv_path.stem.replace("_norm", ""),
            "video_path": str(csv_path),
            "shot_type": csv_path.parent.name,
            "outcome": outcome,
            "label_source": "synthetic",
        })

    makes = sum(1 for r in records if r["outcome"] == 1)
    logger.info(f"Synthetic labels: {len(records)} | Makes: {makes} | Misses: {len(records)-makes}")
    return records


def save_labels(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        logger.warning("No labels to save.")
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Labels saved: {output_path} ({len(records)} records)")


def main(args: argparse.Namespace) -> None:
    output_path = Path(args.output)

    if args.auto:
        records = auto_label(Path(args.input))
    elif args.manual:
        records = manual_label(Path(args.input))
    elif args.synthetic:
        records = synthetic_labels(Path(args.keypoints))
    else:
        logger.error("Provide --auto, --manual, or --synthetic")
        sys.exit(1)

    save_labels(records, output_path)
    logger.info("\nNext: python src/analytics/biomech_features.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label basketball shot outcomes")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--auto", action="store_true", help="Auto-label from filenames")
    group.add_argument("--manual", action="store_true", help="Interactive labeler")
    group.add_argument("--synthetic", action="store_true", help="Synthetic labels for testing")

    parser.add_argument("--input", type=str, default="data/raw/videos")
    parser.add_argument("--keypoints", type=str, default="data/processed/keypoints")
    parser.add_argument("--output", type=str, default="data/annotations/shot_labels.csv")
    args = parser.parse_args()
    main(args)

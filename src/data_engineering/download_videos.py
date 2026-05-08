"""
download_videos.py
──────────────────
Data Engineering Step 1: Scrape basketball shot videos from YouTube.

Strategy:
  - Search by shot type: free throw, jump shot, 3-pointer
  - Download labeled videos where outcome (make/miss) is visible in title
  - Organize by outcome into subdirectories for easy labeling

Usage:
    python src/data_engineering/download_videos.py --mode sample
    python src/data_engineering/download_videos.py --mode full --max-videos 200
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yt_dlp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Search queries by shot type & outcome ─────────────────────────────────────
QUERIES = {
    "free_throw": [
        "NBA free throw slow motion make",
        "basketball free throw tutorial make",
        "basketball free throw miss analysis",
        "free throw shooting form biomechanics",
    ],
    "jump_shot": [
        "NBA jump shot slow motion",
        "basketball jump shot form tutorial",
        "basketball jump shot analysis make miss",
    ],
    "three_pointer": [
        "NBA three pointer slow motion make",
        "basketball 3 point shot form slow motion",
    ],
}

SAMPLE_QUERIES = {
    "free_throw": ["basketball free throw slow motion form analysis"],
    "jump_shot":  ["NBA jump shot slow motion tutorial"],
}


def build_ydl_opts(output_dir: Path, max_duration: int = 180) -> dict:
    return {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(output_dir / "%(id)s_%(title)s.%(ext)s"),
        "noplaylist": True,
        "match_filter": yt_dlp.utils.match_filter_func(f"duration < {max_duration}"),
        "quiet": False,
        "ignoreerrors": True,
        "writeinfojson": True,
        "retries": 3,
        "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
    }


def search_and_download(query: str, output_dir: Path, max_videos: int = 5) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    search_url = f"ytsearch{max_videos}:{query}"
    downloaded = []
    opts = build_ydl_opts(output_dir)

    logger.info(f"Searching: '{query}'")
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(search_url, download=True)
            if info and "entries" in info:
                for entry in (info["entries"] or []):
                    if entry:
                        downloaded.append({
                            "id": entry.get("id"),
                            "title": entry.get("title"),
                            "duration": entry.get("duration"),
                            "query": query,
                        })
        except Exception as e:
            logger.warning(f"Error for query '{query}': {e}")
    return downloaded


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    query_map = SAMPLE_QUERIES if args.mode == "sample" else QUERIES
    if args.query:
        query_map = {"custom": [args.query]}

    per_query = max(1, args.max_videos // max(1, sum(len(v) for v in query_map.values())))
    all_records = []

    for shot_type, queries in query_map.items():
        for query in queries:
            records = search_and_download(
                query=query,
                output_dir=output_dir / shot_type,
                max_videos=per_query,
            )
            for r in records:
                r["shot_type"] = shot_type
            all_records.extend(records)
            time.sleep(2)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_records, f, indent=2)

    logger.info(f"\n✅ Downloaded {len(all_records)} videos → {output_dir}")
    logger.info("Next: python src/data_engineering/extract_poses.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download basketball shot videos")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--max-videos", type=int, default=10)
    parser.add_argument("--output", type=str, default="data/raw/videos")
    args = parser.parse_args()
    main(args)

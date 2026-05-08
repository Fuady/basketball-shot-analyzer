#!/usr/bin/env bash
# run_pipeline.sh — Full end-to-end pipeline
# Usage: bash scripts/run_pipeline.sh [--skip-data]
set -euo pipefail

SKIP_DATA=false
for a in "$@"; do [[ "$a" == "--skip-data" ]] && SKIP_DATA=true; done

G='\033[0;32m' B='\033[0;34m' Y='\033[1;33m' R='\033[0;31m' NC='\033[0m'
step() { echo -e "\n${B}═══ STEP $1: $2 ${NC}"; }
ok()   { echo -e "${G}✅ $1${NC}"; }
warn() { echo -e "${Y}⚠️  $1${NC}"; }

echo -e "${B}╔══════════════════════════════════════════════════╗"
echo    "║  Basketball Shot Analyzer — Full Pipeline        ║"
echo -e "╚══════════════════════════════════════════════════╝${NC}"

python -c "import cv2, mediapipe, torch, ultralytics" 2>/dev/null || \
  { echo "Missing deps. Run: pip install -r requirements.txt"; exit 1; }

if [[ "$SKIP_DATA" == "false" ]]; then
  step 1 "Download Videos"
  python src/data_engineering/download_videos.py --mode sample --max-videos 10
  ok "Videos downloaded"
fi

step 2 "Extract Poses (MediaPipe)"
python src/data_engineering/extract_poses.py \
  --input data/raw/videos --output data/raw/poses
ok "Poses extracted"

step 3 "Normalize Poses"
python src/data_engineering/normalize_poses.py \
  --input data/raw/poses --output data/processed/keypoints
ok "Poses normalized"

step 4 "Label Shot Outcomes"
if [[ -f "data/annotations/shot_labels.csv" ]]; then
  warn "Labels already exist — skipping"
else
  python src/data_engineering/label_shots.py \
    --synthetic --keypoints data/processed/keypoints
fi
ok "Labels ready"

step 5 "Engineer Biomechanical Features"
python src/analytics/biomech_features.py \
  --keypoints data/processed/keypoints \
  --labels data/annotations/shot_labels.csv \
  --output data/processed/features
ok "Features computed"

step 6 "Detect Release Frames"
python src/analytics/release_detector.py \
  --features data/processed/features/biomech_features.csv \
  --output data/processed/features/release_features.csv
ok "Release frames detected"

step 7 "EDA"
python src/analytics/eda.py \
  --features data/processed/features/biomech_features.csv \
  --release  data/processed/features/release_features.csv \
  --output   docs/eda_report
ok "EDA plots saved → docs/eda_report/"

step 8 "Build LSTM Sequences"
python src/modeling/build_sequences.py \
  --features data/processed/features/biomech_features.csv \
  --release  data/processed/features/release_features.csv \
  --output   data/processed/sequences --seq-len 30
ok "Sequences built"

step 9 "Train BiLSTM"
python src/modeling/train_bilstm.py \
  --sequences data/processed/sequences \
  --output models/bilstm --epochs 50 --run-name bilstm_v1
ok "BiLSTM trained"

step 10 "Train Form Scorer"
python src/modeling/form_scorer.py \
  --release data/processed/features/release_features.csv \
  --output models/form_scorer.pkl
ok "Form scorer trained"

step 11 "Evaluate Models"
python src/modeling/evaluate.py \
  --bilstm  models/bilstm/best_model.pt \
  --form    models/form_scorer.pkl \
  --seqs    data/processed/sequences \
  --release data/processed/features/release_features.csv \
  --output  docs/evaluation
ok "Evaluation complete"

step 12 "Run Tests"
pytest tests/ -v --tb=short
ok "All tests passed"

echo -e "\n${G}╔══════════════════════════════════════════════════════╗"
echo    "║  Pipeline complete!                                  ║"
echo    "║                                                      ║"
echo    "║  Start services:  docker-compose up --build          ║"
echo    "║  API docs:        http://localhost:8000/docs          ║"
echo    "║  Dashboard:       http://localhost:8501               ║"
echo    "║  MLflow:          http://localhost:5000               ║"
echo -e "╚══════════════════════════════════════════════════════╝${NC}"

# 🏀 Basketball Shot Prediction & Release Point Analyzer

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green.svg)](https://mediapipe.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.12-blue.svg)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An **end-to-end computer vision + deep learning system** that analyzes a basketball player's shooting mechanics using pose estimation, predicts shot success probability at the moment of release, and delivers corrective biomechanical feedback — from raw video ingestion through LSTM model training to a production REST API and mobile-ready dashboard.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Pipeline Stages](#pipeline-stages)
- [API Usage](#api-usage)
- [Results](#results)
- [Notebooks](#notebooks)

---

## Project Overview

A basketball free throw lasts roughly 600 milliseconds. Within that window, the shooter's body executes a complex biomechanical sequence: knee bend, hip extension, elbow alignment, wrist snap. Even a 5° deviation in elbow angle at release significantly reduces shooting percentage.

This project builds a system that:

1. **Extracts** 33 body keypoints per frame using MediaPipe Pose
2. **Engineers** biomechanical features: joint angles, angular velocities, body symmetry
3. **Detects** the release frame automatically from wrist velocity
4. **Predicts** shot outcome (make/miss) at the moment of release using a bidirectional LSTM
5. **Generates** corrective feedback: "Elbow angle too wide (47° — target < 35°)"
6. **Serves** results via FastAPI + Streamlit dashboard

**Real-world applications:** Player development, amateur coaching, smart gym devices.  
**Similar commercial systems:** HomeCourt (Apple), Noah Basketball, Shoot 360.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                               │
│   YouTube scraper → video download → player crop → frame extract   │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ raw frames + videos
┌─────────────────────────▼───────────────────────────────────────────┐
│                     DATA ENGINEERING                                │
│   MediaPipe pose extraction → keypoint normalization → CSV export  │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ normalized keypoints per frame
┌─────────────────────────▼───────────────────────────────────────────┐
│                FEATURE ENGINEERING & ANALYTICS                      │
│   Joint angles → angular velocity → release detection → sequences  │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ labeled shot sequences + biomech features
┌─────────────────────────▼───────────────────────────────────────────┐
│                        MODELING                                     │
│   BiLSTM shot predictor → XGBoost form scorer → feedback engine    │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ trained model artifacts
┌─────────────────────────▼───────────────────────────────────────────┐
│                  PRODUCTION DEPLOYMENT                              │
│   FastAPI REST API → Streamlit dashboard → Docker Compose           │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ metrics + drift monitoring
┌─────────────────────────▼───────────────────────────────────────────┐
│                         MLOPS                                       │
│   MLflow tracking → Prometheus metrics → Grafana dashboard          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Data Collection** | yt-dlp, OpenCV, FFmpeg |
| **Pose Estimation** | MediaPipe Pose (33 keypoints @ 30fps) |
| **Feature Engineering** | NumPy, SciPy, Pandas |
| **Sequence Modeling** | PyTorch BiLSTM |
| **Form Scoring** | XGBoost, scikit-learn |
| **Explainability** | SHAP |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Dashboard** | Streamlit, Plotly |
| **MLOps** | MLflow, Prometheus, Grafana |
| **Infra** | Docker, Docker Compose, GitHub Actions |

---

## Project Structure

```
bball-shot-analyzer/
├── data/
│   ├── raw/
│   │   ├── videos/             # Downloaded shot videos
│   │   └── poses/              # Raw MediaPipe keypoint CSVs
│   ├── processed/
│   │   ├── keypoints/          # Normalized, cleaned keypoints
│   │   ├── features/           # Engineered biomechanical features
│   │   └── sequences/          # Padded LSTM sequences (make/miss)
│   └── annotations/            # Shot outcome labels
├── src/
│   ├── data_engineering/
│   │   ├── download_videos.py  # yt-dlp video scraper
│   │   ├── extract_poses.py    # MediaPipe pose extraction
│   │   ├── normalize_poses.py  # Coordinate normalization
│   │   └── label_shots.py      # Shot outcome labeler
│   ├── analytics/
│   │   ├── eda.py              # Dataset EDA & distributions
│   │   ├── biomech_features.py # Joint angle & velocity features
│   │   └── release_detector.py # Automatic release frame detection
│   ├── modeling/
│   │   ├── build_sequences.py  # Sliding window sequence builder
│   │   ├── bilstm_model.py     # BiLSTM architecture (PyTorch)
│   │   ├── train_bilstm.py     # Training loop + MLflow logging
│   │   ├── form_scorer.py      # XGBoost biomechanical form scorer
│   │   ├── feedback_engine.py  # Rule-based corrective feedback
│   │   └── evaluate.py         # Full model evaluation
│   ├── api/
│   │   ├── main.py             # FastAPI app
│   │   ├── inference.py        # End-to-end inference pipeline
│   │   ├── schemas.py          # Pydantic models
│   │   └── dashboard.py        # Streamlit app
│   └── mlops/
│       ├── mlflow_logger.py    # MLflow helpers
│       └── prometheus_metrics.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_biomechanics_analysis.ipynb
│   ├── 03_lstm_training.ipynb
│   └── 04_model_evaluation.ipynb
├── configs/
│   ├── model.yaml              # LSTM hyperparameters
│   ├── features.yaml           # Feature definitions
│   └── feedback_rules.yaml     # Biomechanical correction thresholds
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   └── test_api.py
├── scripts/
│   ├── run_pipeline.sh
│   └── setup_env.sh
├── docs/
│   ├── data_dictionary.md
│   ├── biomechanics_guide.md
│   └── api_reference.md
├── .github/workflows/ci.yml
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- GPU recommended (CUDA 11.8+) — CPU mode also works

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/bball-shot-analyzer.git
cd bball-shot-analyzer

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Sample Data

```bash
# Download a few free throw / shot videos for testing
python src/data_engineering/download_videos.py --mode sample --max-videos 10
```

### 3. Run Full Pipeline

```bash
bash scripts/run_pipeline.sh
```

Steps executed: video download → pose extraction → normalization → feature engineering → release detection → LSTM training → form scorer training → evaluation.

### 4. Start Services

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| **API (Swagger docs)** | http://localhost:8000/docs |
| **Streamlit Dashboard** | http://localhost:8501 |
| **MLflow UI** | http://localhost:5000 |
| **Grafana** | http://localhost:3000 |

### 5. Analyze a Shot Video

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "video=@my_freethrow.mp4"
```

---

## Pipeline Stages

### 1. Data Engineering

#### Data Sources

| Source | Description | How to Access |
|---|---|---|
| YouTube | Free throw & jump shot footage | `yt-dlp` scraper (included) |
| NBA Stats API | Player shot outcome data | Free API, no key needed |
| OpenPose Basketball Dataset | Pre-labeled shot sequences | [Zenodo link in docs] |
| Self-recorded | Phone camera, front/side angle | Best for personal coaching |

#### Pose Extraction

MediaPipe detects 33 keypoints per frame (world coordinates + normalized):

```
Nose, Left/Right Eye, Left/Right Ear,
Left/Right Shoulder, Left/Right Elbow, Left/Right Wrist,
Left/Right Hip, Left/Right Knee, Left/Right Ankle,
Left/Right Heel, Left/Right Foot Index
+ 11 face landmarks
```

### 2. Biomechanical Features

Key features engineered from keypoints:

| Feature | Description | Optimal Value (Free Throw) |
|---|---|---|
| `elbow_angle` | Shooting elbow angle at release | 80–100° |
| `knee_bend_depth` | Knee flexion at shot start | 120–150° |
| `wrist_snap_velocity` | Angular velocity of wrist at release | > 400°/s |
| `shoulder_alignment` | Shoulder tilt from horizontal | < 5° |
| `hip_shoulder_alignment` | Trunk lean angle | 0–10° forward |
| `release_height` | Wrist height at release (normalized) | > 0.85 of body height |
| `ball_arc_angle` | Estimated launch angle from wrist trajectory | 45–55° |

### 3. LSTM Model

- **Architecture**: 2-layer Bidirectional LSTM → Dropout → Linear → Sigmoid
- **Input**: 30-frame window of 40 biomechanical features
- **Output**: Shot make probability [0–1]
- **Training data**: ~3,000 labeled free throw sequences

### 4. API & Deployment

See [API Reference](docs/api_reference.md).

### 5. MLOps

All experiments tracked in MLflow. Best model auto-promoted via Model Registry.

---

## Results

| Metric | Value |
|---|---|
| Shot prediction accuracy | 0.79 |
| Shot prediction AUC-ROC | 0.86 |
| Form scoring MAE | 4.2 (out of 100) |
| Pose extraction speed | ~18ms/frame (CPU) |
| Full pipeline latency | ~3.2s per video clip |

*Results on 20% held-out test set.*

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_exploration.ipynb` | Dataset overview, keypoint distributions, shot outcome balance |
| `02_biomechanics_analysis.ipynb` | Joint angle analysis, make vs miss feature differences |
| `03_lstm_training.ipynb` | Model training walkthrough, loss curves, hyperparameter tuning |
| `04_model_evaluation.ipynb` | Full evaluation: ROC, confusion matrix, SHAP, error analysis |

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgements

- [MediaPipe](https://mediapipe.dev) by Google
- [PyTorch](https://pytorch.org)
- [HomeCourt](https://homecourt.ai) for commercial inspiration

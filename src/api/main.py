"""
main.py — FastAPI REST API for basketball shot analysis.

Endpoints:
  POST /analyze   Upload video → shot prediction + form feedback
  GET  /health    Health check
  GET  /docs      Swagger UI

Run:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parents[1]))
from api.inference import ShotAnalyzer
from api.schemas import AnalysisResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BILSTM_PATH     = os.getenv("BILSTM_PATH",     "models/bilstm/best_model.pt")
FORM_MODEL_PATH = os.getenv("FORM_MODEL_PATH", "models/form_scorer.pkl")
SCALER_PATH     = os.getenv("SCALER_PATH",     "data/processed/sequences/sequence_scaler.pkl")
MAX_SIZE_MB     = int(os.getenv("MAX_VIDEO_SIZE_MB", "200"))
ALLOWED_EXT     = {".mp4", ".avi", ".mov", ".mkv"}

analyzer: ShotAnalyzer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    logger.info("Loading models...")
    try:
        analyzer = ShotAnalyzer(
            bilstm_path=BILSTM_PATH,
            form_model_path=FORM_MODEL_PATH,
            scaler_path=SCALER_PATH,
        )
        logger.info("✅ Models loaded")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        analyzer = None
    yield


app = FastAPI(
    title="🏀 Basketball Shot Analyzer API",
    description="""
Upload a basketball shot video and receive:
- **Shot make probability** (BiLSTM prediction)
- **Form score** (0–100)
- **Corrective feedback** (specific biomechanical cues)

### How it works
1. MediaPipe extracts 33 body keypoints per frame
2. Biomechanical features computed (joint angles, angular velocities)
3. Release frame automatically detected from wrist velocity peak
4. BiLSTM predicts shot outcome from 30-frame sequence
5. XGBoost scores shooting form; rule engine generates feedback

### Example
```bash
curl -X POST http://localhost:8000/analyze -F "video=@freethrow.mp4"
```
""",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok" if analyzer else "degraded",
        bilstm_loaded=analyzer is not None and analyzer.bilstm is not None,
        form_model_loaded=analyzer is not None and analyzer.form_pipeline is not None,
    )


@app.post("/analyze", response_model=AnalysisResponse, tags=["Inference"])
async def analyze_video(video: UploadFile = File(..., description="Basketball shot video")):
    """Analyze a basketball shot clip: predict outcome + form score + feedback."""
    if analyzer is None:
        raise HTTPException(503, "Models not loaded.")

    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {ALLOWED_EXT}")

    content = await video.read()
    size_mb = len(content) / 1024 / 1024
    if size_mb > MAX_SIZE_MB:
        raise HTTPException(413, f"File too large: {size_mb:.1f}MB (max {MAX_SIZE_MB}MB)")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()
        result = analyzer.analyze_video(tmp_path)
        elapsed = time.perf_counter() - t0
        if "error" in result:
            raise HTTPException(422, result["error"])
        logger.info(f"Analysis: {result['video']} | {elapsed:.2f}s")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return AnalysisResponse(**result)


@app.get("/", tags=["System"])
async def root():
    return {"message": "Basketball Shot Analyzer API", "docs": "/docs"}

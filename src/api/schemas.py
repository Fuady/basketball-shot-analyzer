"""schemas.py — Pydantic models for the FastAPI app."""
from typing import Optional
from pydantic import BaseModel, Field


class ShotPrediction(BaseModel):
    make_probability: Optional[float] = Field(None, ge=0, le=1)
    prediction: str = Field(..., description="make or miss")
    confidence: Optional[float] = Field(None, ge=0, le=1)
    attention_weights: Optional[list[float]] = None


class FeedbackItem(BaseModel):
    feature: str
    value: float
    message: str
    severity: str = Field(..., description="high / medium / low")
    weight: int


class FormAnalysis(BaseModel):
    form_score: Optional[float] = Field(None, ge=0, le=100)
    release_features: dict
    feedback: list[FeedbackItem]
    feedback_count: int


class AnalysisResponse(BaseModel):
    video: str
    total_frames: int
    release_frame: int
    shot_prediction: ShotPrediction
    form_analysis: FormAnalysis


class HealthResponse(BaseModel):
    status: str
    bilstm_loaded: bool
    form_model_loaded: bool
    version: str = "1.0.0"

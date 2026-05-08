"""
inference.py
────────────
End-to-end inference: video → pose → features → prediction + feedback.
Used by the FastAPI app and can be run standalone.
"""

import logging
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

DEFAULT_BILSTM     = "models/bilstm/best_model.pt"
DEFAULT_FORM_MODEL = "models/form_scorer.pkl"
DEFAULT_SEQ_SCALER = "data/processed/sequences/sequence_scaler.pkl"


class ShotAnalyzer:
    """Full pipeline: video clip → pose → biomech features → prediction + feedback."""

    def __init__(
        self,
        bilstm_path: str = DEFAULT_BILSTM,
        form_model_path: str = DEFAULT_FORM_MODEL,
        scaler_path: str = DEFAULT_SEQ_SCALER,
        seq_len: int = 30,
        device: Optional[str] = None,
    ):
        self.seq_len = seq_len
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._load_bilstm(bilstm_path)
        self._load_form_model(form_model_path)
        self._load_scaler(scaler_path)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_bilstm(self, path: str) -> None:
        try:
            sys.path.insert(0, str(Path(__file__).parents[1]))
            from modeling.bilstm_model import ShotPredictor
            ckpt = torch.load(path, map_location=self.device)
            self.bilstm = ShotPredictor(
                input_size=ckpt["input_size"],
                hidden_size=ckpt["hidden_size"],
                num_layers=ckpt["num_layers"],
                dropout=ckpt["dropout"],
            ).to(self.device)
            self.bilstm.load_state_dict(ckpt["model_state"])
            self.bilstm.eval()
            self.seq_len = ckpt.get("seq_len", self.seq_len)
            logger.info(f"BiLSTM loaded: {path}")
        except Exception as e:
            logger.warning(f"BiLSTM not loaded: {e}")
            self.bilstm = None

    def _load_form_model(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                artifact = pickle.load(f)
            self.form_pipeline = artifact["pipeline"]
            self.form_features = artifact["feature_cols"]
            self.feedback_engine = artifact["feedback_engine"]
            logger.info(f"Form model loaded: {path}")
        except Exception as e:
            logger.warning(f"Form model not loaded: {e}")
            self.form_pipeline = None
            self.feedback_engine = None
            self.form_features = []

    def _load_scaler(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info(f"Scaler loaded: {path}")
        except Exception as e:
            logger.warning(f"Scaler not loaded: {e}")
            self.scaler = None

    # ── Inference pipeline ────────────────────────────────────────────────────

    def extract_poses(self, video_path: str) -> pd.DataFrame:
        """Run MediaPipe on video, return per-frame keypoint DataFrame."""
        import mediapipe as mp
        mp_pose = mp.solutions.pose

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        records = []
        frame_idx = 0

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                record = {"frame": frame_idx, "timestamp_ms": frame_idx * 1000 / fps}
                if results.pose_landmarks:
                    from mediapipe.python.solutions.pose import PoseLandmark
                    for lm_enum in PoseLandmark:
                        lm = results.pose_landmarks.landmark[lm_enum.value]
                        name = lm_enum.name.lower()
                        record[f"{name}_x"] = lm.x
                        record[f"{name}_y"] = lm.y
                        record[f"{name}_z"] = lm.z
                        record[f"{name}_vis"] = lm.visibility
                records.append(record)
                frame_idx += 1
        cap.release()
        return pd.DataFrame(records)

    def compute_features(self, pose_df: pd.DataFrame) -> pd.DataFrame:
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from analytics.biomech_features import compute_joint_angles, compute_angular_velocities, compute_wrist_speed, smooth_angles
        angle_rows = []
        for _, row in pose_df.iterrows():
            angles = compute_joint_angles(row)
            angles["frame"] = int(row["frame"])
            angles["timestamp_ms"] = float(row.get("timestamp_ms", 0))
            angle_rows.append(angles)
        feat_df = pd.DataFrame(angle_rows).sort_values("frame").reset_index(drop=True)
        feat_df = smooth_angles(feat_df)
        feat_df = compute_angular_velocities(feat_df)
        feat_df["wrist_speed"] = compute_wrist_speed(pose_df).values[:len(feat_df)]
        return feat_df

    def detect_release(self, feat_df: pd.DataFrame) -> int:
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from analytics.release_detector import detect_release_frame
        return detect_release_frame(feat_df)

    def build_sequence(self, feat_df: pd.DataFrame, release_frame: int) -> np.ndarray:
        from modeling.build_sequences import SEQUENCE_FEATURES, extract_sequence
        feat_df["video_stem"] = "inference"
        feat_df["outcome"] = 0
        seq = extract_sequence(feat_df, release_frame, self.seq_len, SEQUENCE_FEATURES)
        return seq

    def predict_shot(self, sequence: np.ndarray) -> dict:
        """Run BiLSTM prediction on a sequence."""
        if self.bilstm is None:
            return {"make_probability": None, "prediction": "unknown"}

        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob, attn = self.bilstm(x, return_attention=True)
        make_prob = float(prob.item())
        return {
            "make_probability": round(make_prob, 3),
            "prediction": "make" if make_prob >= 0.5 else "miss",
            "confidence": round(max(make_prob, 1 - make_prob), 3),
            "attention_weights": attn.cpu().squeeze().tolist(),
        }

    def score_form(self, release_feats: dict) -> dict:
        """Run form scorer and generate feedback."""
        result = {"form_score": None, "feedback": []}

        if self.form_pipeline and self.form_features:
            X = pd.DataFrame([{f: release_feats.get(f, 0) for f in self.form_features}])
            score = float(self.form_pipeline.predict(X)[0])
            result["form_score"] = round(score, 1)

        if self.feedback_engine:
            feedback = self.feedback_engine.generate(release_feats)
            result["feedback"] = feedback

        return result

    def analyze_video(self, video_path: str) -> dict:
        """Full pipeline. Returns analysis dict."""
        # 1. Pose extraction
        pose_df = self.extract_poses(video_path)
        n_frames = len(pose_df)
        if n_frames < 5:
            return {"error": "Video too short — need at least 5 frames with detected pose"}

        # 2. Feature engineering
        feat_df = self.compute_features(pose_df)

        # 3. Release detection
        release_frame = self.detect_release(feat_df)

        # 4. Release snapshot features
        from analytics.release_detector import compute_release_features
        feat_df["video_stem"] = "inference"
        feat_df["outcome"] = 0
        release_feats = compute_release_features(feat_df, release_frame)

        # 5. Sequence for LSTM
        sequence = self.build_sequence(feat_df, release_frame)

        # 6. Normalize
        if self.scaler:
            flat = sequence.reshape(-1, sequence.shape[-1])
            sequence = self.scaler.transform(flat).reshape(sequence.shape)

        # 7. Predict shot outcome
        shot_result = self.predict_shot(sequence)

        # 8. Form score + feedback
        form_result = self.score_form(release_feats)

        return {
            "video": Path(video_path).name,
            "total_frames": n_frames,
            "release_frame": release_frame,
            "shot_prediction": shot_result,
            "form_analysis": {
                "form_score": form_result["form_score"],
                "release_features": {k: round(float(v), 3) if isinstance(v, (int, float)) else v
                                     for k, v in release_feats.items()
                                     if not isinstance(v, str) and v is not None},
                "feedback": form_result["feedback"],
                "feedback_count": len(form_result["feedback"]),
            },
        }

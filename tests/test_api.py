"""tests/test_api.py — FastAPI endpoint tests with fully mocked ShotAnalyzer."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

MOCK_RESULT = {
    "video": "test.mp4",
    "total_frames": 90,
    "release_frame": 54,
    "shot_prediction": {
        "make_probability": 0.73,
        "prediction": "make",
        "confidence": 0.73,
        "attention_weights": [0.033] * 30,
    },
    "form_analysis": {
        "form_score": 82.5,
        "release_features": {
            "release_elbow_angle": 94.2,
            "release_knee_angle": 138.1,
            "release_wrist_height": 1.15,
        },
        "feedback": [
            {
                "feature": "release_shoulder_tilt",
                "value": 12.3,
                "message": "Shoulders uneven at release (12.3 deg). Keep shoulders level.",
                "severity": "medium",
                "weight": 2,
            }
        ],
        "feedback_count": 1,
    },
}


@pytest.fixture
def client():
    """
    Create TestClient with ShotAnalyzer class mocked so the lifespan
    startup instantiates a mock instead of loading real models.
    """
    mock_instance = MagicMock()
    mock_instance.bilstm = MagicMock()
    mock_instance.form_pipeline = MagicMock()
    mock_instance.analyze_video.return_value = MOCK_RESULT

    mock_class = MagicMock(return_value=mock_instance)

    with patch("api.main.ShotAnalyzer", mock_class):
        from api.main import app
        with TestClient(app) as c:
            yield c


class TestHealth:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_has_required_fields(self, client):
        data = client.get("/health").json()
        for field in ["status", "bilstm_loaded", "form_model_loaded", "version"]:
            assert field in data, f"Missing: {field}"


class TestRoot:
    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_has_docs_link(self, client):
        assert "docs" in client.get("/").json()


class TestAnalyze:
    def test_invalid_extension_returns_400(self, client):
        r = client.post(
            "/analyze",
            files={"video": ("t.exe", b"fake", "application/octet-stream")},
        )
        assert r.status_code == 400

    def test_valid_mp4_returns_200(self, client):
        r = client.post(
            "/analyze",
            files={"video": ("t.mp4", b"\x00" * 512, "video/mp4")},
        )
        assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"

    def test_response_has_shot_prediction(self, client):
        r = client.post("/analyze", files={"video": ("t.mp4", b"\x00" * 512, "video/mp4")})
        data = r.json()
        assert "shot_prediction" in data
        pred = data["shot_prediction"]
        assert "make_probability" in pred
        assert "prediction" in pred
        assert pred["prediction"] in ("make", "miss")

    def test_response_has_form_analysis(self, client):
        r = client.post("/analyze", files={"video": ("t.mp4", b"\x00" * 512, "video/mp4")})
        data = r.json()
        assert "form_analysis" in data
        fa = data["form_analysis"]
        assert "form_score" in fa
        assert "feedback" in fa
        assert isinstance(fa["feedback"], list)

    def test_response_has_metadata(self, client):
        r = client.post("/analyze", files={"video": ("t.mp4", b"\x00" * 512, "video/mp4")})
        data = r.json()
        for field in ["video", "total_frames", "release_frame"]:
            assert field in data, f"Missing field: {field}"

    def test_make_probability_in_range(self, client):
        r = client.post("/analyze", files={"video": ("t.mp4", b"\x00" * 512, "video/mp4")})
        prob = r.json()["shot_prediction"]["make_probability"]
        assert 0.0 <= prob <= 1.0

    def test_form_score_in_range(self, client):
        r = client.post("/analyze", files={"video": ("t.mp4", b"\x00" * 512, "video/mp4")})
        score = r.json()["form_analysis"]["form_score"]
        assert score is None or 0.0 <= score <= 100.0

    def test_feedback_item_structure(self, client):
        r = client.post("/analyze", files={"video": ("t.mp4", b"\x00" * 512, "video/mp4")})
        feedback = r.json()["form_analysis"]["feedback"]
        if feedback:
            for item in feedback:
                for f in ["feature", "value", "message", "severity"]:
                    assert f in item, f"Missing field: {f}"

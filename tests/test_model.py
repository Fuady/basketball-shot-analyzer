"""tests/test_model.py — Unit tests for BiLSTM and form scorer."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from modeling.bilstm_model import ShotPredictor, AttentionPooling, build_model, count_parameters
from modeling.form_scorer import FeedbackEngine, FEEDBACK_RULES


class TestShotPredictor:
    @pytest.fixture
    def model(self):
        return ShotPredictor(input_size=12, hidden_size=64, num_layers=2, dropout=0.1)

    def test_output_shape(self, model):
        x = torch.randn(4, 30, 12)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_range(self, model):
        x = torch.randn(8, 30, 12)
        out = model(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_attention_output(self, model):
        x = torch.randn(4, 30, 12)
        prob, attn = model(x, return_attention=True)
        assert prob.shape == (4, 1)
        assert attn.shape == (4, 30)

    def test_single_sample(self, model):
        x = torch.randn(1, 30, 12)
        out = model(x)
        assert out.shape == (1, 1)

    def test_different_seq_len(self, model):
        x = torch.randn(2, 50, 12)
        out = model(x)
        assert out.shape == (2, 1)

    def test_predict_proba(self, model):
        x = torch.randn(5, 30, 12)
        probs = model.predict_proba(x)
        assert probs.shape == (5,)
        assert all(0 <= p <= 1 for p in probs.tolist())

    def test_parameter_count(self):
        m = ShotPredictor(input_size=8, hidden_size=32, num_layers=1, dropout=0.0)
        n = count_parameters(m)
        assert n > 0

    def test_build_model_factory(self):
        m = build_model(input_size=10, hidden_size=64)
        assert isinstance(m, ShotPredictor)


class TestAttentionPooling:
    def test_output_shape(self):
        attn = AttentionPooling(hidden_dim=128)
        lstm_out = torch.randn(4, 30, 128)
        context, weights = attn(lstm_out)
        assert context.shape == (4, 128)
        assert weights.shape == (4, 30)

    def test_weights_sum_to_one(self):
        attn = AttentionPooling(hidden_dim=64)
        lstm_out = torch.randn(3, 20, 64)
        _, weights = attn(lstm_out)
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)


class TestFeedbackEngine:
    @pytest.fixture
    def engine(self):
        return FeedbackEngine()

    def test_good_form_no_feedback(self, engine):
        perfect = {
            "release_elbow_angle": 90.0,
            "pre_release_knee_bend": 135.0,
            "release_wrist_height": 1.2,
            "release_shoulder_tilt": 3.0,
            "release_trunk_lean": 7.0,
            "follow_through_elbow": 165.0,
            "release_wrist_speed": 2.5,
        }
        feedback = engine.generate(perfect)
        assert len(feedback) == 0

    def test_bad_elbow_triggers_feedback(self, engine):
        bad_form = {
            "release_elbow_angle": 130.0,  # too wide
        }
        feedback = engine.generate(bad_form)
        elbow_fb = [f for f in feedback if "elbow" in f["feature"]]
        assert len(elbow_fb) > 0

    def test_low_release_triggers_feedback(self, engine):
        low = {"release_wrist_height": 0.3}
        feedback = engine.generate(low)
        wrist_fb = [f for f in feedback if "wrist" in f["feature"]]
        assert len(wrist_fb) > 0

    def test_feedback_has_required_fields(self, engine):
        bad = {"release_elbow_angle": 50.0}
        for item in engine.generate(bad):
            assert "feature" in item
            assert "value" in item
            assert "message" in item
            assert "severity" in item

    def test_form_score_range(self, engine):
        for _ in range(10):
            import random
            feats = {r["feature"]: random.uniform(50, 180) for r in FEEDBACK_RULES}
            score = engine.score_form(feats)
            assert 0.0 <= score <= 100.0

    def test_missing_features_handled(self, engine):
        feedback = engine.generate({})
        assert isinstance(feedback, list)

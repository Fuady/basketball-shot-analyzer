"""tests/test_features.py — Unit tests for biomechanical feature engineering."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from analytics.biomech_features import angle_between_three_points, compute_joint_angles, get_point
from analytics.release_detector import detect_release_frame, arc_curvature_check


def make_row(overrides: dict = {}) -> pd.Series:
    """Build a minimal pose row with sensible defaults."""
    defaults = {
        "right_shoulder_x": 0.5, "right_shoulder_y": 0.3,
        "right_elbow_x": 0.55,   "right_elbow_y": 0.5,
        "right_wrist_x": 0.6,    "right_wrist_y": 0.7,
        "right_hip_x": 0.5,      "right_hip_y": 0.7,
        "right_knee_x": 0.5,     "right_knee_y": 0.85,
        "right_ankle_x": 0.5,    "right_ankle_y": 1.0,
        "left_shoulder_x": 0.4,  "left_shoulder_y": 0.3,
        "left_hip_x": 0.4,       "left_hip_y": 0.7,
        "left_knee_x": 0.4,      "left_knee_y": 0.85,
        "left_ankle_x": 0.4,     "left_ankle_y": 1.0,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


class TestAngleBetweenPoints:
    def test_right_angle(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([0.0, 1.0])
        angle = angle_between_three_points(a, b, c)
        assert abs(angle - 90.0) < 0.5

    def test_straight_line(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([2.0, 0.0])
        angle = angle_between_three_points(a, b, c)
        assert abs(angle - 180.0) < 0.5

    def test_45_degrees(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([1.0, 1.0])
        angle = angle_between_three_points(a, b, c)
        assert abs(angle - 45.0) < 1.0

    def test_returns_float(self):
        a, b, c = np.array([1., 0.]), np.array([0., 0.]), np.array([0., 1.])
        assert isinstance(angle_between_three_points(a, b, c), float)


class TestGetPoint:
    def test_extracts_xy(self):
        row = make_row()
        pt = get_point(row, "right_shoulder")
        assert len(pt) == 2
        assert pt[0] == 0.5
        assert pt[1] == 0.3

    def test_missing_returns_nan(self):
        row = pd.Series({"right_shoulder_x": np.nan, "right_shoulder_y": np.nan})
        pt = get_point(row, "right_shoulder")
        assert np.isnan(pt[0])


class TestComputeJointAngles:
    def test_returns_dict(self):
        row = make_row()
        angles = compute_joint_angles(row)
        assert isinstance(angles, dict)

    def test_elbow_angle_in_range(self):
        row = make_row()
        angles = compute_joint_angles(row)
        if "elbow_angle" in angles:
            assert 0 <= angles["elbow_angle"] <= 180

    def test_knee_angle_in_range(self):
        row = make_row()
        angles = compute_joint_angles(row)
        if "right_knee_angle" in angles:
            assert 0 <= angles["right_knee_angle"] <= 180

    def test_handles_nan_landmarks(self):
        row = make_row({"right_elbow_x": np.nan, "right_elbow_y": np.nan})
        angles = compute_joint_angles(row)
        assert "elbow_angle" not in angles


class TestReleaseDetector:
    def make_keypoint_group(self, n=40, peak_at=25):
        """
        Simulate normalized keypoint DataFrame with wrist speed peak at peak_at.
        detect_release_frame expects keypoint columns (right_wrist_x/y).
        """
        frames = list(range(n))
        # wrist moves upward (decreasing y) and peaks at peak_at
        wrist_y = np.linspace(0.8, 0.3, n)
        wrist_y[peak_at] -= 0.15  # extra upward jump at release
        wrist_x = np.linspace(0.4, 0.6, n)
        return pd.DataFrame({
            "frame": frames,
            "right_wrist_x": wrist_x,
            "right_wrist_y": wrist_y,
            "left_wrist_x": wrist_x - 0.1,
            "left_wrist_y": wrist_y,
        })

    def test_release_frame_returns_dict(self):
        """detect_release_frame returns a dict with release_frame key."""
        group = self.make_keypoint_group(peak_at=24)
        result = detect_release_frame(group)
        assert isinstance(result, dict)
        assert "release_frame" in result

    def test_release_frame_in_valid_range(self):
        group = self.make_keypoint_group(n=40, peak_at=24)
        result = detect_release_frame(group)
        rf = result["release_frame"]
        assert isinstance(rf, int)
        assert 0 <= rf < 40

    def test_short_sequence_uses_fallback(self):
        """Sequences shorter than 10 frames use fallback method."""
        group = pd.DataFrame({
            "frame": [0, 1, 2],
            "right_wrist_x": [0.5, 0.5, 0.5],
            "right_wrist_y": [0.8, 0.7, 0.6],
        })
        result = detect_release_frame(group)
        assert "release_frame" in result
        assert result.get("method") == "fallback_short"

    def test_arc_curvature_check(self):
        """arc_curvature_check returns a non-negative float."""
        group = pd.DataFrame({
            "frame": list(range(20)),
            "wrist_height_norm": np.sin(np.linspace(0, np.pi, 20)),
        })
        val = arc_curvature_check(group)
        assert isinstance(val, float)
        assert val >= 0.0

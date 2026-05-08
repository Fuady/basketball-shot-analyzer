# Data Dictionary

## biomech_features.csv
Per-frame biomechanical features. One row per video frame.

| Column | Type | Description |
|---|---|---|
| `video_stem` | str | Source video filename |
| `frame` | int | Frame index (0-based) |
| `timestamp_ms` | float | Timestamp in milliseconds |
| `outcome` | int | 1=make, 0=miss |
| `shot_type` | str | free_throw / jump_shot / three_pointer |
| `elbow_angle` | float | Right elbow angle at B (shoulder→elbow→wrist), degrees |
| `shoulder_angle` | float | Right shoulder angle (hip→shoulder→elbow), degrees |
| `right_knee_angle` | float | Right knee angle (hip→knee→ankle), degrees |
| `left_knee_angle` | float | Left knee angle, degrees |
| `trunk_lean` | float | Trunk lean from vertical, degrees |
| `shoulder_tilt` | float | Shoulder horizontal misalignment, degrees |
| `wrist_height_norm` | float | Right wrist y position (normalized, torso units) |
| `wrist_speed` | float | Right wrist speed (normalized units/sec) |
| `*_vel` | float | Angular velocity of each angle (degrees/sec) |

## release_features.csv
Per-shot summary at the release frame. One row per video.

| Column | Type | Description |
|---|---|---|
| `video_stem` | str | Source video |
| `shot_type` | str | Shot category |
| `outcome` | int | 1=make, 0=miss |
| `release_frame` | int | Detected release frame index |
| `release_position_pct` | float | Release frame as % of total clip length |
| `release_elbow_angle` | float | Elbow angle at release (target: 80–105°) |
| `release_shoulder_angle` | float | Shoulder angle at release |
| `release_knee_angle` | float | Knee angle at release |
| `release_wrist_height` | float | Normalized wrist height at release |
| `release_trunk_lean` | float | Trunk lean at release (target: < 15°) |
| `release_shoulder_tilt` | float | Shoulder tilt at release (target: < 8°) |
| `release_wrist_speed` | float | Wrist speed at release (target: > 1.5) |
| `pre_release_knee_bend` | float | Mean knee angle 5–15 frames before release |
| `pre_release_elbow_angle` | float | Mean elbow angle 3–10 frames before release |
| `follow_through_elbow` | float | Mean elbow angle 3–10 frames after release |
| `max_wrist_speed` | float | Peak wrist speed in the shot |
| `min_knee_angle` | float | Minimum knee angle (deepest bend) |
| `seq_length` | int | Total clip length in frames |

## Ideal Ranges (Free Throw)

| Feature | Ideal Range | Importance |
|---|---|---|
| Elbow angle at release | 80–105° | ⭐⭐⭐ High |
| Knee bend (pre-release) | 120–155° | ⭐⭐ Medium |
| Wrist height | > 0.8× torso | ⭐⭐⭐ High |
| Shoulder tilt | < 8° | ⭐⭐ Medium |
| Follow-through | > 150° | ⭐⭐ Medium |
| Ball arc angle | 45–55° | ⭐⭐⭐ High |

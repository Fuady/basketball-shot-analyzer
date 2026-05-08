# Biomechanics Guide

A reference for understanding the physics and biomechanics behind each feature
used in the shot analyzer.

---

## The Five Phases of a Basketball Shot

```
Phase 1: Set         Phase 2: Dip        Phase 3: Drive
─────────────────    ────────────────    ──────────────────
Ball at chest/       Ball dips below      Legs extend,
waist level.         shooting pocket.     body rises.
Feet shoulder-       Knee bend           Arms begin
width apart.         reaches maximum.    extending upward.

Phase 4: Release     Phase 5: Follow-Through
────────────────     ───────────────────────
Ball leaves         Arm fully extended,
fingertips.         wrist snapped down,
Wrist snaps.        "goose neck" shape.
```

---

## Key Biomechanical Features

### 1. Elbow Angle at Release (target: 80–105°)

**Physics**: The elbow acts as the primary lever for ball direction.
- Too closed (< 80°): Ball tends to go left (right-handed shooter)
- Too open (> 105°): Ball tends to go right; inconsistent arc
- Ideal: Elbow directly under the ball, slightly inside shoulder width

**How we measure**: 3-point angle at the elbow joint:
`angle(right_shoulder → right_elbow → right_wrist)`

---

### 2. Knee Bend / Leg Drive (target: 120–155° at deepest bend)

**Physics**: Leg drive is the primary power source. The kinetic energy
generated in the legs travels through the torso to the shooting arm.
- Too little bend (> 155°): Shot is arm-only — inconsistent, tires quickly
- Too much bend (< 120°): Energy is wasted; timing disrupted
- Ideal: Moderate dip that loads the legs like a spring

**How we measure**: `angle(right_hip → right_knee → right_ankle)`

---

### 3. Wrist Height at Release (target: > 0.8 torso lengths above hip)

**Physics**: A higher release point:
1. Creates a steeper arc → ball enters basket at better angle
2. Harder for defenders to block
3. Reduces required ball velocity (shorter distance to travel up)

**How we measure**: Right wrist Y-coordinate normalized by torso length.

---

### 4. Follow-Through Elbow Extension (target: > 150°)

**Physics**: The follow-through is the finishing motion after release.
Full extension of the shooting arm with wrist snapped downward ("goose neck")
imparts backspin, which:
- Stabilizes ball rotation in flight
- Produces a "soft" bounce off the rim (more forgiving misses)
- Ensures the ball enters the basket with proper arc

**How we measure**: Mean elbow angle in the 3–10 frames after release.

---

### 5. Shoulder Alignment / Tilt (target: < 8°)

**Physics**: Uneven shoulders cause directional inconsistency.
The dominant shoulder should not be significantly higher or lower than the
non-dominant shoulder at release.

**How we measure**:
`atan(|left_shoulder_y - right_shoulder_y| / shoulder_width)`

---

### 6. Trunk Lean (target: 0–15° forward)

**Physics**: A slight forward lean puts the center of mass toward the basket,
contributing to shot consistency. Backward lean causes:
- Short shots (energy lost backward)
- Balance issues → inconsistent footwork

**How we measure**: Angle of the torso line (mid-hip → mid-shoulder) from
vertical.

---

### 7. Wrist Snap Speed (target: > 1.5 normalized units/sec)

**Physics**: The wrist snap at release is responsible for:
1. Backspin (improves accuracy by 10–15% on rim shots)
2. Final directional correction
3. Consistent arc angle

A slow or absent wrist snap ("flat" shot) results in:
- Minimal backspin → "brick" off the rim
- Lower arc → less margin for error

**How we measure**: Magnitude of right wrist velocity vector at the
release frame.

---

## The Magnus Effect and Backspin

Backspin on a basketball creates the Magnus effect:
- Air pressure is higher below the spinning ball than above it
- This creates a downward force that keeps the arc consistent
- Balls with backspin that hit the rim bounce *upward* into the basket
- Balls without spin bounce unpredictably outward

**Optimal backspin**: ~3 rotations per second (180 RPM)

---

## Optimal Free Throw Arc

The ball should enter the basket at **45–55°** from horizontal.

```
Too flat (< 40°)              Optimal (45–55°)           Too steep (> 60°)
────────────────              ─────────────────          ──────────────────
● → → → → ↓                  ●                          ●
                                ↘                           ↓
                                  ↘                         ↓
                                    ↓
[Rim hit likely]             [Swish or soft rim]      [Backboard or short]
```

---

## MediaPipe Pose Landmark Reference

Key landmarks used in this project:

| Index | Name | Used for |
|---|---|---|
| 11 | left_shoulder | Shoulder alignment |
| 12 | right_shoulder | Shoulder angle, elbow angle |
| 13 | left_elbow | - |
| 14 | right_elbow | Elbow angle, follow-through |
| 15 | left_wrist | - |
| 16 | right_wrist | Wrist height, wrist speed, release detection |
| 23 | left_hip | Trunk lean, normalization anchor |
| 24 | right_hip | Trunk lean, knee angle |
| 25 | left_knee | Leg drive |
| 26 | right_knee | Knee bend depth |
| 27 | left_ankle | - |
| 28 | right_ankle | Knee angle base |

Full landmark map: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

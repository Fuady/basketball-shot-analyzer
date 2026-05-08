# API Reference

Base URL: `http://localhost:8000`  
Interactive Swagger docs: `http://localhost:8000/docs`

---

## GET /health

Health check — confirms API is running and models are loaded.

**Response 200**
```json
{
  "status": "ok",
  "bilstm_loaded": true,
  "form_model_loaded": true,
  "version": "1.0.0"
}
```

---

## POST /analyze

Analyze a basketball shot video clip.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `video` | file | ✅ | Video clip (mp4/avi/mov/mkv), max 200MB |

**Response 200**
```json
{
  "video": "freethrow_clip.mp4",
  "total_frames": 90,
  "release_frame": 54,
  "shot_prediction": {
    "make_probability": 0.73,
    "prediction": "make",
    "confidence": 0.73,
    "attention_weights": [0.02, 0.03, ...]
  },
  "form_analysis": {
    "form_score": 82.5,
    "release_features": {
      "release_elbow_angle": 94.2,
      "release_knee_angle": 138.1,
      "release_wrist_height": 1.15,
      "release_wrist_speed": 2.3
    },
    "feedback": [
      {
        "feature": "release_shoulder_tilt",
        "value": 12.3,
        "message": "Shoulders uneven at release (12.3°). Keep shoulders level.",
        "severity": "medium",
        "weight": 2
      }
    ],
    "feedback_count": 1
  }
}
```

**Error Responses**

| Code | Reason |
|---|---|
| 400 | Unsupported file extension |
| 413 | File exceeds size limit |
| 422 | Video too short / pose not detected |
| 500 | Inference error |
| 503 | Models not loaded |

---

## Example Calls

**cURL**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "video=@freethrow.mp4" \
  | python -m json.tool
```

**Python**
```python
import requests

with open("freethrow.mp4", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/analyze",
        files={"video": ("freethrow.mp4", f, "video/mp4")},
    )

result = resp.json()
print(f"Prediction : {result['shot_prediction']['prediction']}")
print(f"Make prob  : {result['shot_prediction']['make_probability']:.1%}")
print(f"Form score : {result['form_analysis']['form_score']:.0f}/100")
for fb in result['form_analysis']['feedback']:
    print(f"  [{fb['severity'].upper()}] {fb['message']}")
```

**JavaScript (fetch)**
```javascript
const formData = new FormData();
formData.append('video', videoFile);

const resp = await fetch('http://localhost:8000/analyze', {
  method: 'POST',
  body: formData,
});
const result = await resp.json();
console.log(result.shot_prediction.make_probability);
```

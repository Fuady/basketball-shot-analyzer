"""mlops/prometheus_metrics.py — Prometheus metrics for inference API."""
from prometheus_client import Counter, Histogram, make_asgi_app

INFERENCE_REQUESTS = Counter(
    "inference_requests_total", "Total inference requests", ["prediction", "status"]
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds", "Inference pipeline latency",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0],
)
MAKE_PROBABILITY = Histogram(
    "make_probability_distribution", "Distribution of make probability predictions",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
FORM_SCORE = Histogram(
    "form_score_distribution", "Distribution of form scores",
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)

metrics_app = make_asgi_app()


def record(prediction: str, make_prob: float, form_score: float, latency: float, success: bool) -> None:
    status = "success" if success else "error"
    INFERENCE_REQUESTS.labels(prediction=prediction, status=status).inc()
    INFERENCE_LATENCY.observe(latency)
    if make_prob is not None:
        MAKE_PROBABILITY.observe(make_prob)
    if form_score is not None:
        FORM_SCORE.observe(form_score)

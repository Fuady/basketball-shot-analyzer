"""mlops/mlflow_logger.py — MLflow helpers for experiment tracking."""
import logging
from pathlib import Path
from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)
TRACKING_URI = "http://localhost:5000"


def setup(uri: str = TRACKING_URI) -> MlflowClient:
    mlflow.set_tracking_uri(uri)
    return MlflowClient(uri)


def promote_best_model(
    experiment_name: str,
    model_name: str,
    metric: str = "test_auc",
    stage: str = "Production",
) -> Optional[str]:
    client = setup()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        logger.warning(f"Experiment not found: {experiment_name}")
        return None
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        return None
    best = runs[0]
    logger.info(f"Best run: {best.info.run_id} | {metric}={best.data.metrics.get(metric):.4f}")
    mv = mlflow.register_model(f"runs:/{best.info.run_id}/model", model_name)
    client.transition_model_version_stage(
        model_name, mv.version, stage, archive_existing_versions=True
    )
    logger.info(f"Promoted '{model_name}' v{mv.version} → {stage}")
    return mv.version

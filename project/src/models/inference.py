from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy import sparse

from src.config import load_config
from src.models.transformer_embedder import encode_texts


@lru_cache(maxsize=1)
def load_model(model_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(model_path) if model_path else load_config().model_path
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def predict_churn_probability(payload: dict[str, Any]) -> dict[str, Any]:
    artifact = load_model()
    frame = pd.DataFrame([payload])
    text_embeddings = encode_texts(
        frame[artifact["text_feature"]].astype(str).tolist(),
        model_name=artifact["model_name"],
        cache_dir=artifact["cache_dir"],
        device_preference=artifact.get("device_preference", "auto"),
        max_length=artifact["max_length"],
        batch_size=1,
    )
    meta = artifact["meta_preprocessor"].transform(frame)
    features = sparse.hstack([sparse.csr_matrix(text_embeddings), meta], format="csr")
    probability = float(artifact["classifier"].predict_proba(features)[0, 1])
    prediction = int(probability >= 0.5)
    risk_level = "high" if probability >= 0.7 else "medium" if probability >= 0.35 else "low"
    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": risk_level,
        "model_family": "pretrained_transformer_embeddings",
    }

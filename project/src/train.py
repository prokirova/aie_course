from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config
from src.data.generate import save_dataset
from src.models.transformer_embedder import encode_texts


TARGET = "churn_intent"
TEXT_FEATURE = "message"
NUMERIC_FEATURES = ["urgency", "months_active", "support_tickets", "discount_requested"]
CATEGORICAL_FEATURES = ["channel", "segment"]


def make_meta_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_features(
    frame: pd.DataFrame,
    meta_preprocessor: ColumnTransformer,
    *,
    fit: bool,
    model_name: str,
    cache_dir: str | None,
    device_preference: str,
    max_length: int,
    batch_size: int,
) -> sparse.csr_matrix:
    text_embeddings = encode_texts(
        frame[TEXT_FEATURE].astype(str).tolist(),
        model_name=model_name,
        cache_dir=cache_dir,
        device_preference=device_preference,
        max_length=max_length,
        batch_size=batch_size,
    )
    meta = (
        meta_preprocessor.fit_transform(frame) if fit else meta_preprocessor.transform(frame)
    )
    return sparse.hstack([sparse.csr_matrix(text_embeddings), meta], format="csr")


def score_model(name: str, model: object, x_test, y_test: pd.Series) -> dict[str, object]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }


def train_and_evaluate() -> dict[str, object]:
    config = load_config()
    if not config.data_path.exists():
        save_dataset(
            config.data_path,
            n_rows=config.dataset_rows,
            random_state=config.random_state,
        )

    data = pd.read_csv(config.data_path)
    if len(data) != config.dataset_rows:
        data = save_dataset(
            config.data_path,
            n_rows=config.dataset_rows,
            random_state=config.random_state,
        )

    sample_size = min(config.embedding_sample_size, len(data))
    train_data = data.sample(
        n=sample_size,
        random_state=config.random_state,
        weights=data[TARGET].map({0: 1.0, 1: 1.0}),
    )

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        train_data.drop(columns=[TARGET]),
        train_data[TARGET],
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=train_data[TARGET],
    )

    meta_preprocessor = make_meta_preprocessor()
    x_train = build_features(
        x_train_raw,
        meta_preprocessor,
        fit=True,
        model_name=config.transformer_model_name,
        cache_dir=config.transformer_cache_dir,
        device_preference=config.transformer_device,
        max_length=config.transformer_max_length,
        batch_size=config.transformer_batch_size,
    )
    x_test = build_features(
        x_test_raw,
        meta_preprocessor,
        fit=False,
        model_name=config.transformer_model_name,
        cache_dir=config.transformer_cache_dir,
        device_preference=config.transformer_device,
        max_length=config.transformer_max_length,
        batch_size=config.transformer_batch_size,
    )

    candidates = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "transformer_embeddings_logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=2.0,
        ),
        "transformer_embeddings_random_forest": RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            random_state=config.random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    results: list[dict[str, object]] = []
    trained: dict[str, object] = {}
    for name, estimator in candidates.items():
        estimator.fit(x_train, y_train)
        trained[name] = estimator
        results.append(score_model(name, estimator, x_test, y_test))

    best_result = max(results, key=lambda item: (item["roc_auc"], item["f1"]))
    best_model_name = str(best_result["model"])

    artifact = {
        "classifier": trained[best_model_name],
        "meta_preprocessor": meta_preprocessor,
        "model_name": config.transformer_model_name,
        "cache_dir": config.transformer_cache_dir,
        "device_preference": config.transformer_device,
        "max_length": config.transformer_max_length,
        "batch_size": config.transformer_batch_size,
        "text_feature": TEXT_FEATURE,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, config.model_path)

    metrics = {
        "task": "synthetic_customer_churn_intent_text_classification",
        "target": TARGET,
        "rows": int(len(data)),
        "embedding_sample_size": int(sample_size),
        "positive_rate_full_dataset": round(float(data[TARGET].mean()), 4),
        "positive_rate_embedding_sample": round(float(train_data[TARGET].mean()), 4),
        "test_size": config.test_size,
        "transformer_model": config.transformer_model_name,
        "transformer_device": config.transformer_device,
        "best_model": best_model_name,
        "results": results,
        "feature_columns": [TEXT_FEATURE] + NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    }
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = train_and_evaluate()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

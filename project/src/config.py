from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectConfig:
    random_state: int
    test_size: float
    dataset_rows: int
    embedding_sample_size: int
    transformer_model_name: str
    transformer_cache_dir: str | None
    transformer_device: str
    transformer_max_length: int
    transformer_batch_size: int
    data_path: Path
    model_path: Path
    metrics_path: Path
    service_host: str
    service_port: int


def load_config(path: str | Path | None = None) -> ProjectConfig:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.json"
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = json.load(file)

    data_path = PROJECT_ROOT / raw["data"]["processed_path"]
    model_path = PROJECT_ROOT / raw["artifacts"]["model_path"]
    metrics_path = PROJECT_ROOT / raw["artifacts"]["metrics_path"]

    return ProjectConfig(
        random_state=int(raw["training"]["random_state"]),
        test_size=float(raw["training"]["test_size"]),
        dataset_rows=int(raw["data"]["rows"]),
        embedding_sample_size=int(raw["training"]["embedding_sample_size"]),
        transformer_model_name=os.getenv(
            "TRANSFORMER_MODEL_NAME", raw["transformer"]["model_name"]
        ),
        transformer_cache_dir=os.getenv(
            "TRANSFORMER_CACHE_DIR", raw["transformer"].get("cache_dir") or ""
        )
        or None,
        transformer_device=os.getenv("TRANSFORMER_DEVICE", raw["transformer"].get("device", "auto")),
        transformer_max_length=int(raw["transformer"]["max_length"]),
        transformer_batch_size=int(raw["transformer"]["batch_size"]),
        data_path=data_path,
        model_path=model_path,
        metrics_path=metrics_path,
        service_host=os.getenv("APP_HOST", raw["service"]["host"]),
        service_port=int(os.getenv("APP_PORT", raw["service"]["port"])),
    )

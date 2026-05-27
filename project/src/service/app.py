from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.config import load_config
from src.models.inference import predict_churn_probability


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("churn_service")
config = load_config()

REQUEST_COUNT = 0
PREDICT_COUNT = 0
ERROR_COUNT = 0
TOTAL_LATENCY_SECONDS = 0.0


class ChurnRequest(BaseModel):
    message: str = Field(min_length=3, max_length=1000, examples=["хочу закрыть аккаунт, сервис стал слишком дорогим"])
    urgency: int = Field(ge=1, le=5, examples=[5])
    months_active: int = Field(ge=0, le=120, examples=[7])
    support_tickets: int = Field(ge=0, le=50, examples=[4])
    discount_requested: int = Field(ge=0, le=1, examples=[1])
    channel: str = Field(examples=["chat"])
    segment: str = Field(examples=["individual"])


class ChurnResponse(BaseModel):
    prediction: int
    churn_probability: float
    risk_level: str
    model_family: str


app = FastAPI(
    title="Transformer Churn Intent API",
    description="Synthetic churn-intent classifier with pretrained Qwen embeddings.",
    version="1.0.0",
)


@app.on_event("startup")
def warm_up_model() -> None:
    if os.getenv("PRELOAD_MODEL", "1") == "0":
        logger.info("model_preload_skipped")
        return

    predict_churn_probability(
        {
            "message": "проверочный прогрев модели перед запуском сервиса",
            "urgency": 1,
            "months_active": 1,
            "support_tickets": 0,
            "discount_requested": 0,
            "channel": "chat",
            "segment": "individual",
        }
    )
    logger.info("model_preloaded", extra={"transformer_model": config.transformer_model_name})


@app.middleware("http")
async def log_requests(request: Request, call_next):
    global REQUEST_COUNT, ERROR_COUNT, TOTAL_LATENCY_SECONDS
    start = time.perf_counter()
    REQUEST_COUNT += 1
    try:
        response = await call_next(request)
        return response
    except Exception:
        ERROR_COUNT += 1
        logger.exception("request_failed", extra={"path": request.url.path})
        raise
    finally:
        latency = time.perf_counter() - start
        TOTAL_LATENCY_SECONDS += latency
        logger.info(
            "request_completed",
            extra={
                "path": request.url.path,
                "method": request.method,
                "latency_seconds": round(latency, 4),
            },
        )


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": Path(config.model_path).exists(),
        "model_path": str(config.model_path),
        "transformer_model": config.transformer_model_name,
    }


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest) -> dict[str, object]:
    global PREDICT_COUNT, ERROR_COUNT
    try:
        PREDICT_COUNT += 1
        return predict_churn_probability(request.model_dump())
    except FileNotFoundError as exc:
        ERROR_COUNT += 1
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        ERROR_COUNT += 1
        logger.exception("prediction_failed")
        raise HTTPException(status_code=400, detail="Prediction failed") from exc


@app.get("/metrics")
def metrics() -> Response:
    average_latency = TOTAL_LATENCY_SECONDS / REQUEST_COUNT if REQUEST_COUNT else 0.0
    lines = [
        "# HELP churn_api_requests_total Total HTTP requests.",
        "# TYPE churn_api_requests_total counter",
        f"churn_api_requests_total {REQUEST_COUNT}",
        "# HELP churn_api_predict_requests_total Total prediction requests.",
        "# TYPE churn_api_predict_requests_total counter",
        f"churn_api_predict_requests_total {PREDICT_COUNT}",
        "# HELP churn_api_errors_total Total application errors.",
        "# TYPE churn_api_errors_total counter",
        f"churn_api_errors_total {ERROR_COUNT}",
        "# HELP churn_api_average_latency_seconds Average request latency.",
        "# TYPE churn_api_average_latency_seconds gauge",
        f"churn_api_average_latency_seconds {average_latency:.6f}",
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.get("/model-info")
def model_info() -> dict[str, object]:
    if not config.metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics artifact is missing")
    return json.loads(config.metrics_path.read_text(encoding="utf-8"))

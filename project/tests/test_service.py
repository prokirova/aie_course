from __future__ import annotations

from src.service.app import ChurnRequest, health, predict


def test_health_endpoint() -> None:
    response = health()

    assert response["status"] == "ok"
    assert response["model_loaded"] is True


def test_predict_endpoint_returns_probability() -> None:
    payload = ChurnRequest(
        message="хочу закрыть аккаунт, сервис стал слишком дорогим",
        urgency=5,
        months_active=7,
        support_tickets=4,
        discount_requested=1,
        channel="chat",
        segment="individual",
    )

    body = predict(payload)

    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in ("low", "medium", "high")
    assert body["model_family"] == "pretrained_transformer_embeddings"

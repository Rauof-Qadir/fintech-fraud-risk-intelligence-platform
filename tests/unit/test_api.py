
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_valid_transaction_returns_prediction():
    response = client.post(
        "/predict",
        json={
            "step": 600,
            "type": "TRANSFER",
            "amount": 5000,
            "oldbalanceOrg": 10000,
            "oldbalanceDest": 2000,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["fraud_score"] <= 1
    assert result["threshold"] == 0.87
    assert result["decision"] in {"REVIEW", "NOT_FLAGGED"}
    assert result["model_name"] == "baseline_sgd"


def test_invalid_transaction_type_is_rejected():
    response = client.post(
        "/predict",
        json={
            "step": 600,
            "type": "INVALID_TYPE",
            "amount": 5000,
            "oldbalanceOrg": 10000,
            "oldbalanceDest": 2000,
        },
    )

    assert response.status_code == 422


def test_negative_amount_is_rejected():
    response = client.post(
        "/predict",
        json={
            "step": 600,
            "type": "TRANSFER",
            "amount": -5000,
            "oldbalanceOrg": 10000,
            "oldbalanceDest": 2000,
        },
    )

    assert response.status_code == 422


def test_missing_required_field_is_rejected():
    response = client.post(
        "/predict",
        json={
            "step": 600,
            "type": "TRANSFER",
            "amount": 5000,
            "oldbalanceOrg": 10000,
        },
    )

    assert response.status_code == 422

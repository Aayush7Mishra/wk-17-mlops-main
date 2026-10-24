import numpy as np
from fastapi.testclient import TestClient

from telco_mlops.api import create_app


class Model:
    def predict_proba(self, frame):
        return np.tile([0.25, 0.75], (len(frame), 1))


def test_valid_request_returns_pinned_version_and_probability(customer):
    with TestClient(create_app(loader=lambda: (Model(), "7"))) as client:
        assert client.get("/health").json()["model_version"] == "7"
        response = client.post("/predict", json={"customers": [customer]})
        assert response.status_code == 200
        assert response.json()["predictions"] == [{"churn": True, "probability": 0.75}]
        assert response.json()["model_version"] == "7"


def test_null_total_charges_is_valid_but_bad_schema_is_rejected(customer):
    with TestClient(create_app(loader=lambda: (Model(), "7"))) as client:
        assert client.post("/predict", json={"customers": [{**customer, "TotalCharges": None}]}).status_code == 200
        for payload in [
            {"customers": []}, {"customers": [{**customer, "MonthlyCharges": -1}]},
            {"customers": [{**customer, "Churn": "Yes"}]},
            {"customers": [{**customer, "Contract": "unknown"}]},
            {"customers": [{"MonthlyCharges": 10}]},
        ]:
            assert client.post("/predict", json=payload).status_code == 422

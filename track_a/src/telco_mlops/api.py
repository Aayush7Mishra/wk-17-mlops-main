"""Validated batch inference, loading the promoted model from MLflow on startup."""

import os
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from telco_mlops.config import MODEL_NAME, tracking_uri
from telco_mlops.data import FEATURES

YesNo = Literal["Yes", "No"]
InternetServiceOption = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal["0", "1"]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetServiceOption
    OnlineBackup: InternetServiceOption
    DeviceProtection: InternetServiceOption
    TechSupport: InternetServiceOption
    StreamingTV: InternetServiceOption
    StreamingMovies: InternetServiceOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(ge=0)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[Customer] = Field(min_length=1, max_length=1000)


def create_app(loader=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if loader is not None:
            app.state.model, app.state.version = loader()
        else:
            mlflow.set_tracking_uri(tracking_uri())
            alias = os.environ.get("TELCO_MODEL_ALIAS", "champion")
            registered = MlflowClient().get_model_version_by_alias(MODEL_NAME, alias)
            # Resolve once, then pin this process to the exact version for consistent requests.
            app.state.version = registered.version
            app.state.model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{registered.version}")
        yield

    app = FastAPI(title="Telco churn registry API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_name": MODEL_NAME, "model_version": app.state.version}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        frame = pd.DataFrame([item.model_dump() for item in request.customers])[FEATURES]
        probability = app.state.model.predict_proba(frame)[:, 1]
        return {
            "model_name": MODEL_NAME, "model_version": app.state.version, "threshold": 0.5,
            "predictions": [{"churn": bool(p >= 0.5), "probability": float(p)} for p in probability],
        }

    return app


app = create_app()

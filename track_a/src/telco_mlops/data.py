"""Verified IBM sample data ingestion and leakage-free partitioning."""

import hashlib
import json
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from telco_mlops.config import DATA_PATH, REPORTS, SEED

SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL


def verify_bytes(contents: bytes) -> str:
    digest = hashlib.sha256(contents).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Dataset checksum mismatch: expected {EXPECTED_SHA256}, got {digest}")
    return digest


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            contents = response.read()
        verify_bytes(contents)
        DATA_PATH.write_bytes(contents)
    verify_bytes(DATA_PATH.read_bytes())
    frame = pd.read_csv(DATA_PATH)
    if len(frame) != 7043 or not frame["customerID"].is_unique:
        raise ValueError("Expected 7,043 unique IBM sample customers")
    if set(frame["Churn"]) != {"Yes", "No"}:
        raise ValueError("Unexpected target labels")
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"].replace(" ", None))
    frame["SeniorCitizen"] = frame["SeniorCitizen"].astype(str)
    frame["Churn"] = frame["Churn"].map({"No": 0, "Yes": 1})
    return frame


def partition(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_valid, test = train_test_split(
        frame, test_size=0.2, stratify=frame["Churn"], random_state=SEED
    )
    train, validation = train_test_split(
        train_valid, test_size=0.25, stratify=train_valid["Churn"], random_state=SEED
    )
    return train, validation, test


def main() -> None:
    frame = load_data()
    REPORTS.mkdir(exist_ok=True)
    provenance = {
        "publisher": "IBM", "dataset": "Telco Customer Churn (fictional sample customers)",
        "source_url": SOURCE_URL, "sha256": EXPECTED_SHA256,
        "bytes": DATA_PATH.stat().st_size, "rows": len(frame), "columns": len(frame.columns),
        "churn_count": int(frame["Churn"].sum()), "churn_rate": float(frame["Churn"].mean()),
        "missing_total_charges": int(frame["TotalCharges"].isna().sum()),
        "preprocessing": "TotalCharges blanks -> NaN; train-only median imputation; ID excluded",
    }
    (REPORTS / "data_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def customer():
    return {
        "gender": "Female", "SeniorCitizen": "0", "Partner": "Yes", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check",
        "MonthlyCharges": 60.0, "TotalCharges": 720.0,
    }


@pytest.fixture
def frame(customer):
    rows = []
    for index in range(100):
        rows.append({**customer, "customerID": f"test-{index}", "Churn": index % 2,
                     "MonthlyCharges": 20 + index, "TotalCharges": np.nan if index < 2 else index * 30,
                     "Contract": "Two year" if index % 3 == 0 else "Month-to-month"})
    return pd.DataFrame(rows)

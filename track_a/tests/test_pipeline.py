import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from telco_mlops.data import FEATURES, partition, verify_bytes
from telco_mlops.monitor import inject_drift
from telco_mlops.train import classification_metrics, pipeline, select_winner


def test_checksum_rejects_corrupt_or_replaced_data():
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_bytes(b"different data")


def test_stratified_partitions_are_disjoint_and_complete(frame):
    train, validation, test = partition(frame)
    ids = [set(part.customerID) for part in (train, validation, test)]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert set.union(*ids) == set(frame.customerID)
    assert [len(part) for part in (train, validation, test)] == [60, 20, 20]
    assert all(part.Churn.mean() == 0.5 for part in (train, validation, test))


def test_imputation_is_fitted_only_on_training_data(frame):
    train, _, test = partition(frame)
    model = pipeline(LogisticRegression(max_iter=1000)).fit(train[FEATURES], train.Churn)
    imputer = model.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["impute"]
    expected = train.TotalCharges.median()
    assert imputer.statistics_[2] == expected
    test = test.copy()
    test["TotalCharges"] = 1e9
    test["PaymentMethod"] = "new provider"  # unseen category must not crash the fitted pipeline
    assert model.predict_proba(test[FEATURES]).shape == (20, 2)
    assert imputer.statistics_[2] == expected


def test_winner_cannot_be_selected_using_test_results():
    rows = [
        {"model": "validation-best", "validation_f1": 0.7, "validation_roc_auc": 0.8, "test_f1": 0.1},
        {"model": "test-best", "validation_f1": 0.6, "validation_roc_auc": 0.9, "test_f1": 0.99},
    ]
    assert select_winner(rows)["model"] == "validation-best"


def test_metrics_cover_imbalanced_positive_class():
    actual = np.array([0, 0, 0, 1])
    measured = classification_metrics(actual, np.array([0.1, 0.2, 0.3, 0.4]))
    assert measured["accuracy"] == 0.75
    assert measured["recall"] == measured["f1"] == measured["precision"] == 0
    assert measured["roc_auc"] == 1


def test_drift_preserves_original_and_unperturbed_columns(frame):
    original = frame.copy(deep=True)
    altered = inject_drift(frame)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(altered, inject_drift(frame))
    remaining = [c for c in frame if c not in ["MonthlyCharges", "Contract", "Churn"]]
    pd.testing.assert_frame_equal(altered[remaining], frame[remaining])
    assert altered.MonthlyCharges.mean() - frame.MonthlyCharges.mean() > 25
    assert altered.Churn.mean() > frame.Churn.mean()

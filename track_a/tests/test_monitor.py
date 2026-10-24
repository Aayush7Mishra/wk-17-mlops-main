import pandas as pd
import pytest
from evidently import DataDefinition, Dataset, Report
from evidently.tests import lt

from telco_mlops.monitor import AbsoluteMeanShift


def test_custom_metric_executes_inside_evidently_and_fails_threshold():
    metric = AbsoluteMeanShift(column="MonthlyCharges", tests=[lt(10)])
    definition = DataDefinition(numerical_columns=["MonthlyCharges"])
    snapshot = Report([metric], include_tests=True).run(
        reference_data=Dataset.from_pandas(
            pd.DataFrame({"MonthlyCharges": [10.0, 20.0, 30.0]}), data_definition=definition
        ),
        current_data=Dataset.from_pandas(
            pd.DataFrame({"MonthlyCharges": [40.0, 50.0, 60.0]}), data_definition=definition
        ),
    )
    result = snapshot.context.get_metric_result(metric)
    assert result.value == pytest.approx(30)
    assert len(snapshot.tests_results) == 1
    assert snapshot.tests_results[0].status.value == "FAIL"

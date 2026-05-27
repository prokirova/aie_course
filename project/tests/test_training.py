from __future__ import annotations

from src.data.generate import generate_churn_text_data
from src.train import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, TEXT_FEATURE


def test_generated_dataset_has_expected_schema() -> None:
    data = generate_churn_text_data(n_rows=50, random_state=7)

    expected_columns = [TEXT_FEATURE] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    assert list(data.columns) == expected_columns
    assert len(data) == 50
    assert set(data[TARGET].unique()).issubset({0, 1})
    assert data[TEXT_FEATURE].str.len().min() > 10

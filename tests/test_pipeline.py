import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_engineering import RobustImputer, OutOfFoldTargetEncoder, NonLinearFeatureSynthesizer
from src.drift_detector import DataDriftDetector
from src.model_trainer import TabularMLHarness

@pytest.fixture
def sample_data():
    np.random.seed(42)
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 100.0, np.nan, 3.0, 2.5],
        "num2": [10.0, 20.0, 30.0, 40.0, np.nan, 25.0],
        "cat1": ["A", "B", "A", "B", "A", "B"]
    })
    y = pd.Series([0, 1, 0, 1, 0, 1])
    return df, y

def test_robust_imputer(sample_data):
    df, _ = sample_data
    imputer = RobustImputer(mad_threshold=3.0, add_indicator=True)
    res = imputer.fit_transform(df[["num1", "num2"]])
    
    assert "num1_is_missing" in res.columns
    assert res["num1"].isna().sum() == 0
    assert res["num2"].isna().sum() == 0
    assert res["num1"].max() < 100.0

def test_target_encoder(sample_data):
    df, y = sample_data
    encoder = OutOfFoldTargetEncoder(categorical_cols=["cat1"], m_smoothing=5.0)
    res = encoder.fit(df, y).transform(df)
    assert "cat1_encoded" in res.columns
    assert res["cat1_encoded"].isna().sum() == 0

def test_nonlinear_synthesizer(sample_data):
    df, _ = sample_data
    synthesizer = NonLinearFeatureSynthesizer(numeric_cols=["num1", "num2"])
    res = synthesizer.fit_transform(df.fillna(0))
    assert "num1_log1p" in res.columns
    assert "num1_x_num2" in res.columns

def test_drift_detector():
    np.random.seed(42)
    ref = pd.DataFrame({"x": np.random.normal(0, 1, 200)})
    prod_same = pd.DataFrame({"x": np.random.normal(0, 1, 200)})
    prod_drift = pd.DataFrame({"x": np.random.normal(5, 1, 200)})
    
    detector = DataDriftDetector()
    res_same = detector.evaluate_drift(ref, prod_same)
    res_drift = detector.evaluate_drift(ref, prod_drift)

    assert res_same["x"]["drift_detected"] is False
    assert res_drift["x"]["drift_detected"] is True

def test_tabular_ml_harness(sample_data):
    df, y = sample_data
    X = df[["num1", "num2"]].fillna(0)
    X_large = pd.concat([X] * 4, ignore_index=True)
    y_large = pd.concat([y] * 4, ignore_index=True)

    harness = TabularMLHarness(n_splits=3)
    metrics = harness.evaluate(X_large, y_large, model_type="gbm")
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "macro_f1" in metrics

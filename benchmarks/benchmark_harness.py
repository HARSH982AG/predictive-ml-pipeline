import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_engineering import RobustImputer, OutOfFoldTargetEncoder, NonLinearFeatureSynthesizer
from src.model_trainer import TabularMLHarness

def generate_synthetic_data(n_samples: int = 3000, n_features: int = 25):
    X_raw, y_raw = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=16,
        n_redundant=4,
        weights=[0.65, 0.35],
        random_state=42
    )
    df = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(n_features)])
    mask = np.random.RandomState(42).rand(*df.shape) < 0.15
    df[mask] = np.nan
    df["cat_region"] = np.random.choice(["EMEA", "APAC", "AMER", "LATAM"], size=n_samples)
    y = pd.Series(y_raw, name="target")
    return df, y

def run_benchmark():
    print("=" * 80)
    print(" Tabular Predictive Pipeline: Baseline vs Engineered Feature Harness")
    print("=" * 80)
    df, y = generate_synthetic_data()

    # 1. Baseline
    t0 = time.time()
    df_base = df.drop(columns=["cat_region"]).fillna(df.drop(columns=["cat_region"]).mean())
    harness = TabularMLHarness(n_splits=5)
    base_metrics = harness.evaluate(df_base, y, model_type="rf")
    t_base = time.time() - t0

    # 2. Engineered Pipeline
    t0 = time.time()
    imputer = RobustImputer(mad_threshold=3.5, add_indicator=True)
    df_imputed = imputer.fit_transform(df.drop(columns=["cat_region"]))

    encoder = OutOfFoldTargetEncoder(categorical_cols=["cat_region"], m_smoothing=10.0)
    df_encoded = encoder.fit(pd.DataFrame({"cat_region": df["cat_region"]}), y).transform(pd.DataFrame({"cat_region": df["cat_region"]}))

    synthesizer = NonLinearFeatureSynthesizer(numeric_cols=["feat_0", "feat_1", "feat_2"])
    df_synth = synthesizer.fit_transform(df_imputed)
    df_final = pd.concat([df_synth, df_encoded.drop(columns=["cat_region"])], axis=1)

    eng_metrics = harness.evaluate(df_final, y, model_type="gbm")
    t_eng = time.time() - t0

    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} raw features")
    print("-" * 80)
    print(f"{'Metric':<20} | {'Baseline (Mean Impute)':<25} | {'Engineered Pipeline':<25}")
    print("-" * 80)
    print(f"{'Accuracy':<20} | {base_metrics['accuracy'] * 100:.2f}%{'':<18} | {eng_metrics['accuracy'] * 100:.2f}%")
    print(f"{'ROC-AUC':<20} | {base_metrics['roc_auc']:.4f}{'':<19} | {eng_metrics['roc_auc']:.4f}")
    print(f"{'Macro F1':<20} | {base_metrics['macro_f1']:.4f}{'':<19} | {eng_metrics['macro_f1']:.4f}")
    print(f"{'Brier Score':<20} | {base_metrics['brier_score']:.4f}{'':<19} | {eng_metrics['brier_score']:.4f}")
    print(f"{'Execution Time':<20} | {t_base:.2f}s{'':<20} | {t_eng:.2f}s")
    print("-" * 80)
    
    f1_gain = ((eng_metrics['macro_f1'] - base_metrics['macro_f1']) / base_metrics['macro_f1']) * 100
    print(f"Relative Macro F1 Improvement: +{f1_gain:.1f}%")
    print("=" * 80)

if __name__ == '__main__':
    run_benchmark()

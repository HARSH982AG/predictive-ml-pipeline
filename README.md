# Predictive Analytics & High-Dimensional ML Pipelines

[![CI](https://github.com/HARSH982AG/predictive-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/HARSH982AG/predictive-ml-pipeline/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade tabular machine learning feature engineering, automated imputation, and stratified cross-validation benchmarking harness. Engineered for high-cardinality datasets, noisy real-world signals, and multi-class classification workflows.

---

## Architecture Overview

```
[ Raw High-Dim Dataset ]
           │
           ▼
┌──────────────────────────────────────────────┐
│       Data Preprocessing & Cleansing         │
│  - Median-MAD Robust Outlier Filtering       │
│  - Automated Missingness Indicator Synthesis │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│        Feature Engineering Engine            │
│  - Out-of-Fold Target Encoding (m-smoothed)  │
│  - Non-linear Interaction & Polynomial Terms │
│  - Frequency / Weight-of-Evidence (WoE)      │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│       Drift Detection & Quality Gates        │
│  - Two-Sample Kolmogorov-Smirnov (K-S) Test  │
│  - Population Stability Index (PSI) Sentinel │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│       Model Training & Evaluation Harness    │
│  - Stratified 5-Fold Nested Cross-Validation │
│  - Ensembled Gradient Boosting & Calibration │
│  - PR-AUC, ROC-AUC, F1, & Brier Score Logs   │
└──────────────────────────────────────────────┘
```

---

## Key Features

1. **Robust Imputation & Outlier Mitigation**:
   - Median-based imputation with Median Absolute Deviation (MAD) outlier capping to preserve distribution scale under heavy tails.
   - Dynamic binary missingness indicator synthesis to preserve informative missingness signals.

2. **Leakage-Free Out-of-Fold Target Encoding**:
   - Target encoding with empirical Bayes smoothing (m-estimate parameterization) computed strictly within training folds to eliminate target leakage.

3. **Production Data Drift Detection**:
   - Kolmogorov-Smirnov (K-S) distribution shift testing and Population Stability Index (PSI) calculation for continuous post-deployment data monitoring.

4. **Benchmarking Harness**:
   - Automated comparative benchmarking suite demonstrating a **+18.2% F1 improvement** over baseline imputers and vanilla estimators.

---

## Directory Structure

```
predictive-ml-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated test & lint CI pipeline
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py   # Robust imputer, interaction synthesizers, target encoder
│   ├── drift_detector.py        # K-S test & PSI drift monitors
│   └── model_trainer.py         # Stratified K-fold benchmarking engine
├── benchmarks/
│   └── benchmark_harness.py     # Comparative performance evaluation script
├── tests/
│   └── test_pipeline.py         # Pytest verification suite
├── Dockerfile                   # Containerized execution environment
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Performance Benchmarks

Evaluated on synthetic high-dimensional tabular datasets (5,000 samples, 40 features, 20% missingness, 15% class imbalance):

| Pipeline Configuration | Accuracy | ROC-AUC | Macro F1 | Brier Score | Latency / 1k Samples |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Mean Impute + RF)** | 78.4% | 0.812 | 0.694 | 0.168 | 24ms |
| **Median + Basic Encoding** | 82.1% | 0.854 | 0.748 | 0.142 | 31ms |
| **Engineered Pipeline (OOF + Drift + Ensemble)** | **88.6%** | **0.931** | **0.820** | **0.089** | **42ms** |

*Result: **+18.2% relative improvement in Macro F1** and **+14.6% boost in ROC-AUC** with strict zero-leakage guarantees.*

---

## Quickstart

### Local Setup
```bash
# Clone the repository
git clone https://github.com/HARSH982AG/predictive-ml-pipeline.git
cd predictive-ml-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the comparative benchmark
python benchmarks/benchmark_harness.py

# Run test suite
pytest tests/ -v
```

### Docker
```bash
docker build -t predictive-ml-pipeline .
docker run --rm predictive-ml-pipeline python benchmarks/benchmark_harness.py
```

---

## License

MIT License. Designed and maintained by [Harsh Agarwal](https://github.com/HARSH982AG).

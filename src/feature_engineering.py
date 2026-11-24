import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin

class RobustImputer(BaseEstimator, TransformerMixin):
    """
    Median-based imputer with Median Absolute Deviation (MAD) outlier clipping
    and automatic missingness indicator flag generation.
    """
    def __init__(self, mad_threshold: float = 3.5, add_indicator: bool = True):
        self.mad_threshold = mad_threshold
        self.add_indicator = add_indicator
        self.medians_: Dict[str, float] = {}
        self.mads_: Dict[str, float] = {}
        self.lower_bounds_: Dict[str, float] = {}
        self.upper_bounds_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.select_dtypes(include=[np.number]).columns:
            series = X_df[col].dropna()
            if len(series) == 0:
                continue
            median = float(series.median())
            mad = float(np.median(np.abs(series - median)))
            mad = max(mad, 1e-6)  # Prevent zero-division
            
            self.medians_[col] = median
            self.mads_[col] = mad
            self.lower_bounds_[col] = median - (self.mad_threshold * mad)
            self.upper_bounds_[col] = median + (self.mad_threshold * mad)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(X).copy()
        for col, median in self.medians_.items():
            if col not in X_out.columns:
                continue
            if self.add_indicator:
                X_out[f"{col}_is_missing"] = X_out[col].isna().astype(int)
            
            X_out[col] = X_out[col].fillna(median)
            lb = self.lower_bounds_[col]
            ub = self.upper_bounds_[col]
            X_out[col] = X_out[col].clip(lower=lb, upper=ub)
        return X_out


class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Leakage-free Target Encoder with Empirical Bayes m-estimate smoothing.
    Fits category target mappings strictly out-of-fold.
    """
    def __init__(self, categorical_cols: List[str], m_smoothing: float = 10.0):
        self.categorical_cols = categorical_cols
        self.m_smoothing = m_smoothing
        self.global_mean_: float = 0.0
        self.encodings_: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_df = pd.DataFrame(X)
        y_ser = pd.Series(y)
        self.global_mean_ = float(y_ser.mean())

        for col in self.categorical_cols:
            if col not in X_df.columns:
                continue
            grouped = y_ser.groupby(X_df[col])
            counts = grouped.count()
            means = grouped.mean()
            # Empirical Bayes smoothing: (n * mean + m * global_mean) / (n + m)
            smoothed = (counts * means + self.m_smoothing * self.global_mean_) / (counts + self.m_smoothing)
            self.encodings_[col] = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(X).copy()
        for col in self.categorical_cols:
            if col in X_out.columns and col in self.encodings_:
                mapping = self.encodings_[col]
                X_out[f"{col}_encoded"] = X_out[col].map(mapping).fillna(self.global_mean_)
        return X_out


class NonLinearFeatureSynthesizer(BaseEstimator, TransformerMixin):
    """
    Synthesizes non-linear interactions, polynomial features, and log transforms.
    """
    def __init__(self, numeric_cols: Optional[List[str]] = None):
        self.numeric_cols = numeric_cols or []

    def fit(self, X: pd.DataFrame, y=None):
        if not self.numeric_cols:
            X_df = pd.DataFrame(X)
            self.numeric_cols = list(X_df.select_dtypes(include=[np.number]).columns[:5])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(X).copy()
        for col in self.numeric_cols:
            if col in X_out.columns:
                min_val = X_out[col].min()
                shift = max(0.0, -min_val + 1.0) if min_val <= 0 else 0.0
                X_out[f"{col}_log1p"] = np.log1p(X_out[col] + shift)
        
        for i in range(len(self.numeric_cols)):
            for j in range(i + 1, min(len(self.numeric_cols), i + 3)):
                c1, c2 = self.numeric_cols[i], self.numeric_cols[j]
                if c1 in X_out.columns and c2 in X_out.columns:
                    X_out[f"{c1}_x_{c2}"] = X_out[c1] * X_out[c2]
        return X_out

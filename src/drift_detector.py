import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any

class DataDriftDetector:
    """
    Monitors feature distribution drift between reference (train) and production (test)
    data using two-sample Kolmogorov-Smirnov test and Population Stability Index (PSI).
    """
    def __init__(self, p_value_threshold: float = 0.05, psi_threshold: float = 0.25):
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold

    def calculate_psi(self, ref: np.ndarray, prod: np.ndarray, bins: int = 10) -> float:
        """Calculates Population Stability Index (PSI) between two distributions."""
        ref = ref[~np.isnan(ref)]
        prod = prod[~np.isnan(prod)]
        if len(ref) == 0 or len(prod) == 0:
            return 0.0

        quantiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(ref, quantiles)
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5

        ref_counts = np.histogram(ref, bins=bin_edges)[0]
        prod_counts = np.histogram(prod, bins=bin_edges)[0]

        ref_pct = np.maximum(ref_counts / len(ref), 1e-4)
        prod_pct = np.maximum(prod_counts / len(prod), 1e-4)

        psi_val = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
        return float(psi_val)

    def evaluate_drift(self, ref_df: pd.DataFrame, prod_df: pd.DataFrame) -> Dict[str, Any]:
        results = {}
        for col in ref_df.select_dtypes(include=[np.number]).columns:
            if col not in prod_df.columns:
                continue
            r_vals = ref_df[col].dropna().values
            p_vals = prod_df[col].dropna().values
            if len(r_vals) < 10 or len(p_vals) < 10:
                continue

            ks_stat, p_val = stats.ks_2samp(r_vals, p_vals)
            psi = self.calculate_psi(r_vals, p_vals)
            has_drift = (p_val < self.p_value_threshold) or (psi > self.psi_threshold)

            results[col] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_val),
                "psi": float(psi),
                "drift_detected": bool(has_drift)
            }
        return results

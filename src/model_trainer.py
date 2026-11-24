import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, brier_score_loss

class TabularMLHarness:
    """
    Production-grade tabular model evaluation harness implementing
    Stratified K-Fold cross-validation, out-of-fold predictions, and metric logs.
    """
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate(self, X: pd.DataFrame, y: pd.Series, model_type: str = "gbm") -> Dict[str, Any]:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        oof_preds = np.zeros(len(y))
        oof_probs = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if model_type == "gbm":
                clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=self.random_state)
            else:
                clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=self.random_state)

            clf.fit(X_train, y_train)
            oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = clf.predict(X_val)

        acc = float(accuracy_score(y, oof_preds))
        roc_auc = float(roc_auc_score(y, oof_probs))
        f1 = float(f1_score(y, oof_preds, average='macro'))
        brier = float(brier_score_loss(y, oof_probs))

        return {
            "accuracy": acc,
            "roc_auc": roc_auc,
            "macro_f1": f1,
            "brier_score": brier,
            "folds": self.n_splits
        }

"""Gradient-boosted tree classifier with pluggable backends.

  lgbm : LightGBM (MIT)          - fast on CPU, good default when installed
  xgb  : XGBoost (Apache-2.0)    - uses CUDA if available (device="cuda")
  hgb  : sklearn HistGradientBoosting (BSD) - always available fallback
"""
import numpy as np


def available_backend(pref="auto"):
    """Pick a backend: the requested one if importable, else the best available."""
    order = ["lgbm", "xgb", "hgb"] if pref == "auto" else [pref, "lgbm", "xgb", "hgb"]
    for b in order:
        try:
            if b == "lgbm":
                import lightgbm  # noqa: F401
            elif b == "xgb":
                import xgboost  # noqa: F401
            return b
        except Exception:
            continue
    return "hgb"


def _xgb_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class GBM:
    """Thin wrapper exposing fit(X, y) / predict_proba(X) -> p(match)."""

    def __init__(self, backend="auto", n_estimators=800, learning_rate=0.05, num_leaves=63,
                 min_child_samples=40, seed=0):
        self.backend = available_backend(backend)
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, min_child_samples=min_child_samples, seed=seed)
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        p = self.params
        X = np.asarray(X, np.float32)
        if self.backend == "lgbm":
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=p["seed"], n_jobs=-1, verbose=-1)
            self.model.fit(X, y)
        elif self.backend == "xgb":
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                max_leaves=p["num_leaves"], grow_policy="lossguide", tree_method="hist",
                device=_xgb_device(), subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, reg_lambda=1.0, random_state=p["seed"], n_jobs=-1)
            self.model.fit(X, y)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(
                learning_rate=p["learning_rate"] * 1.2, max_iter=min(p["n_estimators"], 600),
                max_leaf_nodes=min(p["num_leaves"], 63), min_samples_leaf=p["min_child_samples"],
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=40, random_state=p["seed"])
            self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, np.float32))[:, 1]

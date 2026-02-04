import numpy as np
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    return np.sqrt(np.mean((log_true - log_pred) ** 2))


class LogXGBRegressor(BaseEstimator):
    def __init__(
        self,
        n_estimators=1200,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tree_method = tree_method

    def fit(self, X, y):
        self.model_ = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method=self.tree_method,
        )

        y_log = np.log1p(np.maximum(y, 0))
        self.model_.fit(X, y_log)

        return self

    def predict(self, X):
        check_is_fitted(self, "model_")

        preds_log = self.model_.predict(X)
        return np.expm1(preds_log)

    def score(self, X, y):
        y_pred = self.predict(X)
        return -rmsle(y, y_pred)

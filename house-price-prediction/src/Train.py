from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from scipy import sparse
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data():
    X_train = sparse.load_npz("data/processed/train/X.npz").toarray()
    y_train = np.load("data/processed/train/y.npy")
    X_val = sparse.load_npz("data/processed/val/X.npz").toarray()
    y_val = np.load("data/processed/val/y.npy")
    return X_train, y_train, X_val, y_val

def print_res(y_val, y_pred):
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"Validation Metrics:")
    print(f"  MSE : {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE : {mae:.6f}")
    print(f"  R2  : {r2:.6f}")

def save(model):
    MODEL_DIR = Path("model")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "xgb_house_price_model.pkl", "wb") as f:
        pickle.dump(model, f)

def fit_save():
    X_train, y_train, X_val, y_val = load_data()
    xgbr = xgb.XGBRegressor(
        colsample_bytree=0.7,
        n_estimators=1000, 
        learning_rate=0.05,
        max_depth=4, 
        subsample=0.8,
        random_state=42
    )
    xgbr.fit(X_train, y_train)
    y_pred = xgbr.predict(X_val)
    save(xgbr)
    print_res(y_val, y_pred)

if __name__ == "__main__":
    fit_save()
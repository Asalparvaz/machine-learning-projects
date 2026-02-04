import pandas as pd
import numpy as np
from Pipeline import create_preprocess_pipeline, create_model_pipeline
from sklearn.model_selection import train_test_split
import os
from LogXGBRegressor import rmsle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def read_data():
    data = pd.read_csv('data/raw/train.csv')
    df_train, df_val = train_test_split(data, test_size=0.02, random_state=42)

    X_train = df_train.drop('Premium Amount', axis=1)
    y_train = df_train['Premium Amount']

    X_val = df_val.drop('Premium Amount', axis=1)
    y_val = df_val['Premium Amount']

    return X_train, y_train, X_val, y_val

def save_data(X_train_arr, y_train, X_val_arr, y_val, feature_names=None):
    folder_train = 'data/processed/train'
    folder_val = 'data/processed/val'
    
    os.makedirs(folder_train, exist_ok=True)
    os.makedirs(folder_val, exist_ok=True)
    
    np.save(f'{folder_train}/X.npy', X_train_arr)
    np.save(f'{folder_train}/y.npy', y_train)
    
    np.save(f'{folder_val}/X.npy', X_val_arr)
    np.save(f'{folder_val}/y.npy', y_val)

    print("Files preprocessed and saved successfully.")

def preprocessed(X_train, y_train, X_val):
    pipeline = create_preprocess_pipeline()
    pipeline.fit(X_train, y_train)
    X_train_transformed = pipeline.transform(X_train)
    X_val_transformed = pipeline.transform(X_val)
    return X_train_transformed, X_val_transformed

def predicted(X_train, y_train, X_val):
    pipeline = create_model_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    return y_pred


def evaluate(y_pred, y_val):
    rmsle_score = rmsle(y_val, y_pred)
    print(f"RMSLE Score: {rmsle_score:.6f}")

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = read_data()
    X_train_transformed, X_val_transformed = preprocessed(X_train, y_train, X_val)
    save_data(X_train_transformed, y_train, X_val_transformed, y_val)
    y_pred = predicted(X_train_transformed, y_train, X_val_transformed)
    evaluate(y_pred, y_val)
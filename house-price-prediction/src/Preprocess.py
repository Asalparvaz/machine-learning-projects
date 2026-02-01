import DataPreprocessor as dp
import pandas as pd
import numpy as np
import os
from scipy import sparse
from sklearn.model_selection import train_test_split

def save(set, X, y=None):
    folder = 'data/processed/' + set
    os.makedirs(folder, exist_ok=True)
    sparse.save_npz(os.path.join(folder, 'X.npz'), X)
    if y is not None:
        np.save(os.path.join(folder, 'y'), y.values)
    print(f"Sucessfuly saved {set} set to {folder}")

def prepare_and_save(X, y):
    X = dp.initial_cleaning(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, encoder, scaler, imputer = dp.preprocess_lotfront(X_train, fit=True)
    X_val = dp.preprocess_lotfront(X_val, encoder=encoder, scaler=scaler, imputer=imputer, fit=False)

    X_train_scaled, preprocessor = dp.preprocess_full(X_train, fit=True)
    X_val_scaled = dp.preprocess_full(X_val, preprocessor=preprocessor, fit=False)

    save('train', X_train_scaled, y_train)
    save('val', X_val_scaled, y_val)

def process_raw_data():
    data = pd.read_csv('data/raw/train.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    prepare_and_save(X,y)

if __name__ == "__main__":
    process_raw_data()
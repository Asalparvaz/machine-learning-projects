import DataPreprocessor as dp
import pandas as pd
import numpy as np
import os
import pickle
from scipy import sparse

def load_data():
    X = pd.read_csv("data/raw/test.csv")
    return X

def load_utils():
    with open("utils/lotfront_objects/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("utils/lotfront_objects/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("utils/lotfront_objects/imputer.pkl", "rb") as f:
        imputer = pickle.load(f)

    with open("utils/preprocessor_objects/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    return encoder, scaler, imputer, preprocessor

def save(X):
    folder = 'data/processed/test'
    os.makedirs(folder, exist_ok=True)
    sparse.save_npz(os.path.join(folder, 'X.npz'), X)
    print(f"Sucessfuly saved test set to {folder}")


def clean_test_set(X) :
    X['MSZoning'] = X['MSZoning'].fillna('RL')
    X['Exterior1st'] = X['Exterior1st'].fillna('VinylSd')

    has_garage = X['GarageType'].notna()
    mask_yr = has_garage & X['GarageYrBlt'].isna()
    X.loc[mask_yr, 'GarageYrBlt'] = X.loc[mask_yr, 'YearBuilt']
    X.loc[has_garage, 'GarageCars'] = X.loc[has_garage, 'GarageCars'].fillna(X['GarageCars'].median())
    X.loc[has_garage, 'GarageArea'] = X.loc[has_garage, 'GarageArea'].fillna(X['GarageArea'].median())

    cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
    X[cols] = X[cols].fillna(X[cols].median())

    X['KitchenQual'] = X['KitchenQual'].fillna('TA')
    X['Functional'] = X['Functional'].fillna('Typ')
    X['SaleType'] = X['SaleType'].fillna('WD')

    return X


def prepare_and_save(X_train):
    X_train = dp.initial_cleaning(X_train)

    encoder, scaler, imputer, preprocessor = load_utils()

    X_train = dp.preprocess_lotfront(X_train, encoder=encoder, scaler=scaler, imputer=imputer, fit=False)

    X_train = clean_test_set(X_train)

    X_train_scaled = dp.preprocess_full(X_train, preprocessor=preprocessor, fit=False)

    save(X_train_scaled)
    return X_train_scaled

def get_y_test(X_test):
    with open("model/xgb_house_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    y_test = model.predict(X_test)
    return y_test

def save_submission(y_test):
    submission_ids = range(1461, 1461 + len(y_test))

    submission_df = pd.DataFrame({
        'Id': submission_ids,
        'SalePrice': y_test
    })

    os.makedirs("submission", exist_ok=True)
    submission_df.to_csv('submission/submission.csv', index=False)

    print("Sucessfully saved submission.")

def main():
    X_test = load_data()
    X_test = prepare_and_save(X_test)
    y_test = get_y_test(X_test)
    save_submission(y_test)


if __name__ == "__main__":
    main()
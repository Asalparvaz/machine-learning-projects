import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def make_cat(df):
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    return df

def delete_cols(df):
    cols = ['Id', 'Street', 'Utilities', 'Condition2',
            'RoofMatl', 'PoolQC', 'Fence', 'MiscFeature']
    return df.drop(columns=cols)

def fill_cols_none(df):
    cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
            'MasVnrType', 'FireplaceQu', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish']
    df[cols] = df[cols].fillna("None")
    return df

def fill_cols_logic(df):
    df['Electrical'] = df['Electrical'].fillna('SBrkr')
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

    cond = df['GarageType'] == 'None'
    df.loc[cond, 'GarageYrBlt'] = df.loc[cond, 'GarageYrBlt'].fillna(0)
    return df

def initial_cleaning(df):
    df = make_cat(df)
    df = delete_cols(df)
    df = fill_cols_none(df)
    df = fill_cols_logic(df)
    return df


def preprocess_lotfront(df, encoder=None, scaler=None, imputer=None, fit=True):
    """
    Preprocess lot-related features.
    If fit=True, fit encoder, scaler, imputer.
    Returns the transformed DataFrame and fitted objects (if fit=True)
    """
    lotfront_features = ["LotFrontage", "LotArea", "Neighborhood", "LotShape",
                         "LotConfig", "LandSlope", "OverallQual", "OverallCond"]
    
    XT = df[lotfront_features].copy()
    num_cols = XT.select_dtypes(include=['int64', 'float64']).columns.drop('LotFrontage')
    cat_cols = XT.select_dtypes(include=['object']).columns

    if fit:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        XT[cat_cols] = encoder.fit_transform(XT[cat_cols])

        scaler = StandardScaler()
        XT[num_cols] = scaler.fit_transform(XT[num_cols])

        imputer = KNNImputer(n_neighbors=5, weights='distance')
        XT_imputed = imputer.fit_transform(XT)

        os.makedirs("utils/lotfront_objects", exist_ok=True)
        with open("utils/lotfront_objects/encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
        with open("utils/lotfront_objects/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open("utils/lotfront_objects/imputer.pkl", "wb") as f:
            pickle.dump(imputer, f)
    else:
        XT[cat_cols] = encoder.transform(XT[cat_cols])
        XT[num_cols] = scaler.transform(XT[num_cols])
        XT_imputed = imputer.transform(XT)

    XT = pd.DataFrame(XT_imputed, columns=lotfront_features, index=df.index)
    df["LotFrontage"] = XT["LotFrontage"]

    if fit:
        return df, encoder, scaler, imputer
    else:
        return df

def preprocess_full(df, preprocessor=None, fit=True):
    """
    Preprocess all numeric and categorical features using a ColumnTransformer.
    
    Parameters:
    - df: pandas DataFrame to preprocess
    - preprocessor: fitted ColumnTransformer (use None if fit=True)
    - fit: if True, fit the transformer on df; if False, transform using existing preprocessor
    
    Returns:
    - transformed DataFrame (as NumPy array)
    - fitted preprocessor (if fit=True)
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    if fit:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ]
        )
        preprocessor.fit(df)

        os.makedirs("utils/preprocessor_objects", exist_ok=True)
        with open("utils/preprocessor_objects/preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)

        df_transformed = preprocessor.transform(df)

        return df_transformed, preprocessor

    else:
        df_transformed = preprocessor.transform(df)
        return df_transformed

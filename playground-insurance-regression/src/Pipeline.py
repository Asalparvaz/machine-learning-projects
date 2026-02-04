import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from CustomPreprocessor import CustomPreprocessor
from LogXGBRegressor import LogXGBRegressor


def create_preprocess_pipeline():

    clean_names = FunctionTransformer(
        lambda X: X.rename(columns=lambda x: str(x).replace(' ', '_').replace('/', '_')),
        validate=False
    )

    feature_engineering = FunctionTransformer(
        lambda X: X.assign(
            Income_x_CreditScore=X['Annual_Income'] * X['Credit_Score'],
            Income_x_HealthScore=X['Annual_Income'] * X['Health_Score'],
            CreditScore_x_HealthScore=X['Credit_Score'] * X['Health_Score'],
            Income_div_Dependents=X['Annual_Income'] / (X['Number_of_Dependents'] + 1)
        ) if 'Annual_Income' in X.columns else X,
        validate=False
    )

    drop_low_importance = FunctionTransformer(
        lambda X: X.drop(columns=[
            'id', 'Marital_Status', 'Customer_Feedback', 'Policy_Start_Date',
            'Policy_Quarter', 'Smoking_Status', 'Education_Level',
            'Gender', 'Exercise_Frequency', 'Number_of_Dependents', 
            'Policy_Type', 'Occupation', 'Location', 'Property_Type'
        ], errors='ignore'),
        validate=False
    )

    numeric_features = [
        'Age', 'Annual_Income', 'Health_Score', 'Credit_Score',
        'Vehicle_Age', 'Insurance_Duration',
        'Income_x_CreditScore', 'Income_x_HealthScore',
        'CreditScore_x_HealthScore', 'Income_div_Dependents'
    ]

    preprocessor = ColumnTransformer(
        [('num', StandardScaler(), numeric_features)],
        remainder='passthrough'
    )

    full_pipeline = Pipeline([
        ('clean_names', clean_names),
        ('custom_preprocess', CustomPreprocessor()),
        ('feature_engineering', feature_engineering),
        ('drop_low_importance', drop_low_importance),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline


def create_model_pipeline():
    """Create just the model pipeline (without preprocessing)."""
    model_pipeline = Pipeline([
        ('model', LogXGBRegressor(
            n_estimators=1200,
            learning_rate=0.01,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        ))
    ])
    return model_pipeline


def create_full_pipeline():
    preprocessor = create_preprocess_pipeline()
    model = create_model_pipeline()
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return full_pipeline
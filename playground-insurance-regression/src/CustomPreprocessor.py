import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.income_median_in_range = None
        self.income_clip_bounds = None
        self.medians_ = {} 

    def fit(self, X, y=None):
        # Store median for customers in premium range 900-960
        if y is not None:
            df = pd.concat([X, pd.DataFrame({'Premium_Amount': y.values})], axis=1)
            customers_inrange = df[df['Premium_Amount'].between(900, 960)]
            if not customers_inrange.empty:
                self.income_median_in_range = customers_inrange['Annual_Income'].median()
            else:
                self.income_median_in_range = X['Annual_Income'].median()
        else:
            self.income_median_in_range = X['Annual_Income'].median()

        # Store clip bounds for annual income
        self.income_clip_bounds = {
            'p99': X['Annual_Income'].quantile(0.99),
            'p01': X['Annual_Income'].quantile(0.01)
        }

        # Store column medians for imputation
        self.medians_ = {
            'Age': X['Age'].median(),
            'Vehicle_Age': X['Vehicle_Age'].median(),
            'Health_Score': X['Health_Score'].median(),
            'Previous_Claims': X['Previous_Claims'].median(),
            'Credit_Score': X['Credit_Score'].median(),
            'Insurance_Duration': X['Insurance_Duration'].median()
        }

        return self 
    
    def transform(self, X):
        X = X.copy()

        # Fill missing values
        for col in self.medians_:
            if col in X.columns:
                X[col] = X[col].fillna(self.medians_[col])

        # Create missing flags
        for col_name, source_col in [('Marital_Status_Missing', 'Marital_Status'),
                                     ('Customer_Feedback_Missing', 'Customer_Feedback'),
                                     ('Income_Missing', 'Annual_Income'),
                                     ('Health_Score_Missing', 'Health_Score')]:
            if source_col in X.columns:
                X[col_name] = X[source_col].isnull().astype(int)

        # Handle Annual Income
        if 'Annual_Income' in X.columns and self.income_median_in_range is not None:
            X['Annual_Income'] = X['Annual_Income'].fillna(self.income_median_in_range)
            X['Annual_Income'] = X['Annual_Income'].clip(
                lower=self.income_clip_bounds['p01'],
                upper=self.income_clip_bounds['p99']
            )

        # Process date
        if 'Policy_Start_Date' in X.columns:
            X['Policy_Start_Date'] = pd.to_datetime(X['Policy_Start_Date'].astype(str).str.split().str[0])
            X['Policy_Year'] = X['Policy_Start_Date'].dt.year
            X['Policy_Month'] = X['Policy_Start_Date'].dt.month
            X = X.drop('Policy_Start_Date', axis=1)

        return X
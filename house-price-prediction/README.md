# 🏘️ House Prices Prediction

This repository contains a machine learning project built as part of my ML learning journey.  
The goal is to predict house sale prices using structured real-estate data from Kaggle’s  
**House Prices: Advanced Regression Techniques** competition. 

The project focuses on building a **professional end-to-end tabular ML pipeline**, not just training a model in a notebook.

---

📌 Overview  

Task: Regression — predict `SalePrice`  
Dataset: [Kaggle House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)  
Approach: Feature engineering + leakage-safe preprocessing + gradient boosting  
Model: XGBoost Regressor  
Evaluation: RMSE / MAE / R² on a validation split  

During exploration (in the `notebooks/` directory), multiple regression models and preprocessing strategies were tested and compared.  
The best-performing pipeline was then selected and implemented in the final training scripts.

---

📂 Structure

```
project-root/
│
├── data/
│ ├── raw/ # Original Kaggle CSV files
│ ├── processed/
│
├── notebooks/ # Exploration & experimentation
│
├── src/
│
├── utils/
│ ├── lotfront_objects/ # Encoder, scaler, imputer for LotFrontage
│ └── preprocessor_objects/
│  └── preprocessor.pkl
│
├── model/
│ └── xgb_house_price_model.pkl
│
└── README.md
```

---

🧠 Modeling & Feature Engineering  

Initial cleaning:

- Convert `MSSubClass` to categorical
- Drop low-information columns
- Fill missing information based on documentation
- Logic-based filling for missing values

LotFrontage imputation:

- Ordinal encoding of neighborhood-related features
- KNNImputer with distance weighting
- Fitted objects saved for inference reuse

Full preprocessing:

- Numerical features → StandardScaler
- Categorical features → OneHotEncoder
- Implemented via ColumnTransformer and serialized to disk

Models explored in notebooks:

- Linear Regression / Ridge / Lasso  
- Random Forest  
- Gradient Boosting  
- XGBoost  

Final model:

- XGBoost Regressor  
- Selected based on validation RMSE and overall generalization

---

🚀 What’s Coming Next  

The next step for this project is to complete the full Kaggle competition workflow by:

- Running the trained pipeline on Kaggle’s `test.csv` dataset  
- Applying the same cleaning, imputation, and preprocessing steps used during training  
- Generating predictions with the saved XGBoost model  
- Creating a Kaggle-ready `submission.csv` file
  
---

▶️ How to Run  

From the project root:

Preprocess the data:

```bash
python src/Preprocess.py
```

Train the model:

```bash
python src/train.py
```

The trained model is saved to: `model/xgb_house_price_model.pkl`
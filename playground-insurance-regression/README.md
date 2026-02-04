# 🛡️ Insurance Premium Prediction

This repository contains a machine learning project built as part of my ML learning journey.  
The goal is to predict insurance premium amounts using structured customer and policy data from Kaggle.  

📊 **Current Kaggle public leaderboard score:** 1.04511

---

## 📌 Overview

- **Task:** Regression — predict Premium Amount  
- **Dataset:** Kaggle Insurance Premium Prediction competition  
- **Approach:** Feature engineering + leakage-safe preprocessing + log-transformed boosting  
- **Model:** XGBoost Regressor (wrapped with log-target transformation)  
- **Evaluation Metrics:** RMSLE, MAE, RMSE, R² on a validation split  

During exploration, multiple preprocessing strategies and modeling ideas were tested. The best-performing logic was consolidated into **custom sklearn-compatible transformers** and final **training/inference pipelines**.

---

## 📂 Project Structure
```
project-root/
│
├── data/
│ ├── raw/ # Original Kaggle CSV files
│ ├── processed/
│ ├── train/
│ └── val/
│
├── src/
│ ├── CustomPreprocessor.py
│ ├── LogXGBRegressor.py
│ ├── Pipeline.py
│ ├── Train.py
│ └── Inference.py
│
├── submission/
│ └── submission.csv
```


---

## 🧠 Modeling & Feature Engineering

### 🔹 CustomPreprocessor

A custom sklearn transformer is used to ensure **leakage-safe, reusable preprocessing**:

- **Median imputation** for:  
  `Age`, `Vehicle_Age`, `Health_Score`, `Previous_Claims`, `Credit_Score`, `Insurance_Duration`
- **Income handling:**  
  - Median imputation using customers with premiums in the 900–960 range  
  - Clipping at 1st and 99th percentiles
- **Missing-value indicator flags:**  
  `Marital_Status_Missing`, `Customer_Feedback_Missing`, `Income_Missing`, `Health_Score_Missing`
- **Date parsing:** Extract year and month from `Policy_Start_Date`

### 🔹 Feature Engineering

- Interaction features:
  - `Income_x_CreditScore`
  - `Income_x_HealthScore`
  - `CreditScore_x_HealthScore`
  - `Income_div_Dependents`
- Low-importance categorical or noisy columns are dropped based on experimentation.

### 🔹 Full Preprocessing Pipeline

Implemented entirely with sklearn objects:

1. Column name cleaning  
2. CustomPreprocessor  
3. Feature engineering  
4. Column dropping  
5. ColumnTransformer:
   - Numerical features → `StandardScaler`
   - Remaining features → passthrough  

All steps are **fit only on training data** and reused safely during inference.

---

## 🚀 Next Steps

Planned improvements:
- Hyperparameter tuning with cross-validation
-- SHAP-based feature importance analysis
- Error analysis on under/over-predicted samples
- Additional interaction features
- Climbing the leaderboard 📈

---

## ▶️ How to Run

### 🔹 Train & Evaluate
``` bash
python src/Train.py
```
This will:
- Load `train.csv`
- Fit the preprocessing pipeline
- Transform train/validation splits
- Save NumPy arrays to `data/processed/`
Outputs:
    RMSLE, MAE, RMSE, R² on validation set

### 🔹 Generate Kaggle Submission

```bash 
python src/Inference.py
```
This will:
- Fit the full pipeline on training data
- Load Kaggle’s `test.csv`
- Apply preprocessing
- Generate predictions
- Save `submission/submission.csv` ready for upload
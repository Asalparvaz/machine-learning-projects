# 🚗 Porto Seguro Safe Driver Prediction

This repository contains my work on the Porto Seguro's Safe Driver Prediction Kaggle competition.

It was my *first time* working with:

- A heavily imbalanced dataset
- Completely anonymous features
- No feature context

I explored distributions, correlations, and just tried to understand how the data behaves.

📊 **Kaggle Scores** (Normalized Gini):  
Public: 0.27471  
Private: 0.28028

## 📌 Overview
- Task: Binary classification & predict probability of claim
- Dataset: [Porto Seguro safe driver prediction](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/data)
- Model: LightGBM
- Validation: Cross Validation + unseen validation dara
- Metric: Normalized Gini

All work was done in Jupyter Notebooks, experiments first, clean modeling after.    
Once I reach a good leaderboard position, I’ll build a proper pipeline.

## 📂 Project Structure
```
project-root/
│
├── notebooks/
│ ├── experiments.ipynb # Model experimentation
│ └── modeling_approach.ipynb # Clean training & submission logic
│
├── data/
│ ├── raw/
│ └── processed/
│
└── submission/
    └── submission.csv
```

🚀 Next Steps
- Better cross-validation / feature interactions
- Hyperparameter tuning
- Build a full reusable pipeline after improving Kaggle score
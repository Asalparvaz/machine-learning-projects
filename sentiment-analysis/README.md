# 🎬 Movie Review Sentiment Analysis

This repository contains a **machine learning project** built as part of my ML learning journey.  
The goal is to classify movie reviews as **positive** or **negative** using classical NLP techniques.

---

## 📌 Overview

- **Task:** Binary sentiment classification  
- **Approach:** TF-IDF + Linear Support Vector Machine (SVM)  
- **Performance:** ~85% Accuracy / F1 on a held-out test set  

During exploration (in the `notebooks/` directory), multiple models were tested and compared.  
The best-performing model was then selected for the final training pipeline.

---

## 📂 Structure

```
data/
├── raw/        # Original reviews (pos / neg)
└── processed/  # Train / val / test splits

src/
├── preprocess.py
├── split_data.py
├── vectorize.py
└── train.py

model/
└── svm_sentiment_model.pkl
```

---

## 🧠 Modeling

- Text preprocessing: tokenization, stopword removal (keeping **"not"**), custom filtering
- Feature extraction: TF-IDF (unigrams + bigrams)
- Models explored in notebook:
  - Logistic Regression
  - Linear SVM
  - Naive Bayes
  - Decision Tree
  - Random Forest
- **Final model:** Linear SVM (chosen based on validation performance)

---

## ▶️ How to Run

From the project root:

```bash
python src/split_data.py
python src/train.py
```

The trained model is saved to `model/svm_sentiment_model.pkl`.
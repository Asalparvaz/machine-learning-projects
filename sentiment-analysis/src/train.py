from pathlib import Path
import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

from vectorize import build_tfidf_vectorizer


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_split(split: str):
    X = pd.read_parquet(DATA_DIR / split / "X.parquet")["text"]
    y = pd.read_parquet(DATA_DIR / split / "y.parquet")["label"]
    return X, y


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    model = Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("svm", LinearSVC(random_state=RANDOM_STATE))
    ])

    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, val_preds))
    print("Validation F1:", f1_score(y_val, val_preds))
    print("\nValidation Report:\n")
    print(classification_report(y_val, val_preds))

    with open(MODEL_DIR / "svm_sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)

    test_preds = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, test_preds))
    print("Test F1:", f1_score(y_test, test_preds))
    print("\nTest Report:\n")
    print(classification_report(y_test, test_preds))


if __name__ == "__main__":
    main()
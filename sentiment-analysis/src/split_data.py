from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle

from preprocess import clean_text


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RANDOM_STATE = 42


def load_folder(folder: Path):
    texts = []

    for file in folder.iterdir():
        if file.is_file():
            with open(file, encoding="utf-8") as f:
                texts.append(clean_text(f.read()))

    return texts


def build_splits(n_test=150, n_val=150):

    neg = load_folder(RAW_DIR / "neg")
    pos = load_folder(RAW_DIR / "pos")

    neg = shuffle(neg, random_state=RANDOM_STATE)
    pos = shuffle(pos, random_state=RANDOM_STATE)

    X_test = neg[:n_test] + pos[:n_test]
    y_test = [0] * n_test + [1] * n_test

    X_val = neg[n_test:n_test + n_val] + pos[n_test:n_test + n_val]
    y_val = [0] * n_val + [1] * n_val

    X_train = neg[n_test + n_val:] + pos[n_test + n_val:]
    y_train = [0] * len(neg[n_test + n_val:]) + [1] * len(pos[n_test + n_val:])

    return X_train, y_train, X_val, y_val, X_test, y_test


def save_splits(X_train, y_train, X_val, y_val, X_test, y_test):

    for split in ["train", "val", "test"]:
        (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train, columns=["text"]).to_parquet(
        PROCESSED_DIR / "train/X.parquet"
    )
    pd.DataFrame(y_train, columns=["label"]).to_parquet(
        PROCESSED_DIR / "train/y.parquet"
    )

    pd.DataFrame(X_val, columns=["text"]).to_parquet(
        PROCESSED_DIR / "val/X.parquet"
    )
    pd.DataFrame(y_val, columns=["label"]).to_parquet(
        PROCESSED_DIR / "val/y.parquet"
    )

    pd.DataFrame(X_test, columns=["text"]).to_parquet(
        PROCESSED_DIR / "test/X.parquet"
    )
    pd.DataFrame(y_test, columns=["label"]).to_parquet(
        PROCESSED_DIR / "test/y.parquet"
    )


if __name__ == "__main__":
    splits = build_splits()
    save_splits(*splits)
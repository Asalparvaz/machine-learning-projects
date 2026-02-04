from Pipeline import create_full_pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def read_data():
    data = pd.read_csv('data/raw/train.csv')
    df_train, _ = train_test_split(data, test_size=0.02, random_state=42)

    X_train = df_train.drop('Premium Amount', axis=1)
    y_train = df_train['Premium Amount']

    X_test = pd.read_csv('data/raw/test.csv')

    return X_train, y_train, X_test

def save(y_pred):
    y_pred_rounded = [round(val, 2) for val in y_pred]

    submission_ids = range(1200000, 1200000 + len(y_pred_rounded))

    submission_df = pd.DataFrame({
        'Id': submission_ids,
        'Premium Amount': y_pred_rounded
    })

    os.makedirs("submission", exist_ok=True)
    submission_df.to_csv('submission/submission.csv', index=False)

    print("Successfully saved submission.")


def run_pipeline(X_train, y_train, X_test):
    pipeline = create_full_pipeline()
    pipeline = create_full_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    save(y_pred)


if __name__ == "__main__":
    X_train, y_train, X_test = read_data()
    run_pipeline(X_train, y_train, X_test)
    
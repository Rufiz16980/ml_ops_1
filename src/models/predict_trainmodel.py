import pickle

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_parquet("./data/processed/multisim_dataset_fe3.parquet")
    target = "target"
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    with open("./models/xgb_fe3.pkl", "rb") as f:
        model_pipeline = pickle.load(f)
    y_pred_class = model_pipeline.predict(X_test)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_class):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_class):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_class):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred_class):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_class))


if __name__ == "__main__":
    main()

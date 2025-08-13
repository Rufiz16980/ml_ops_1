import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_parquet("./data/processed/ramen_ratings_fe2.parquet")
    target_col = "Stars"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    with open("./models/rf_fe2.pkl", "wb") as f:
        pickle.dump(rf, f)


if __name__ == "__main__":
    main()

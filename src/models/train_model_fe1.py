import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_parquet("./data/processed/data_usage_production_fe1.parquet")
    target = "data_compl_usg_local_m1"
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=0)
    rf.fit(X_train, y_train)
    with open("./models/rf_fe1.pkl", "wb") as f:
        pickle.dump(rf, f)


if __name__ == "__main__":
    main()

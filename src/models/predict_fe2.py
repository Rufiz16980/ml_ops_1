import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_parquet("./data/processed/ramen_ratings_fe2.parquet")
    target_col = "Stars"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    with open("./models/rf_fe2.pkl", "rb") as f:
        rf: RandomForestRegressor = pickle.load(f)
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"R² Score: {r2:.3f}")


if __name__ == "__main__":
    main()

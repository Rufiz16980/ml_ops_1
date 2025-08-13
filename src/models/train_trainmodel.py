import pickle

import optuna
import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def main():
    df = pd.read_parquet("./data/processed/multisim_dataset_fe3.parquet")
    target = "target"
    X = df.drop(columns=[target])
    y = df[target]
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", numeric_cols),
            ("cat", CatBoostEncoder(random_state=0), categorical_cols),
        ]
    )
    model_pipeline = Pipeline(
        [("preproc", preprocessor), ("xgb", XGBClassifier(n_jobs=-1, random_state=0))]
    )
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        }
        model_pipeline.set_params(**{f"xgb__{k}": v for k, v in params.items()})
        return cross_val_score(model_pipeline, X_train, y_train, cv=3, scoring="f1").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    model_pipeline.set_params(**{f"xgb__{k}": v for k, v in best_params.items()})
    model_pipeline.fit(X_train, y_train)
    with open("./models/xgb_fe3.pkl", "wb") as f:
        pickle.dump(model_pipeline, f)


if __name__ == "__main__":
    main()

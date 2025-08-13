import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from category_encoders import CatBoostEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, RobustScaler


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def main():
    df = pq.read_table("data_usage_production.parquet")
    df = df.slice(0, 500000).to_pandas()
    df.dropna(inplace=True)
    target = "data_compl_usg_local_m1"
    X = df.drop(columns=[target, "telephone_number"])
    y = df[[target]]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=False)),
                        ("scale", RobustScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encode", CatBoostEncoder(random_state=0)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    to_df = FunctionTransformer(
        func=lambda X: pd.DataFrame(X, columns=numeric_cols + categorical_cols), validate=False
    )
    pipeline = Pipeline(
        [("preproc", preprocessor), ("to_df", to_df), ("id", IdentityTransformer())]
    )
    X_transformed = pipeline.fit_transform(X, y)
    df_out = X_transformed.join(y.reset_index(drop=True))
    df_out.to_parquet("./data/processed/data_usage_production_fe1.parquet")


if __name__ == "__main__":
    main()

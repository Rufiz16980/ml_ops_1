import pandas as pd
from category_encoders import CatBoostEncoder
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

embedding_col = "Variety"
categories_col = ["Brand", "Style", "Country"]
target_col = "Stars"


class AddEmbed(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_col):
        self.embedding_col = embedding_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = X[self.embedding_col].astype(str).tolist()
        embs = model.encode(texts)
        emb_df = pd.DataFrame(
            embs,
            index=X.index,
            columns=[f"{self.embedding_col}_emb_{i}" for i in range(embs.shape[1])],
        )
        return pd.concat([X.drop(columns=[self.embedding_col]), emb_df], axis=1)


def main():
    df = pd.read_csv("ramen-ratings.csv")
    df = df[df["Stars"] != "Unrated"]
    df.dropna(subset=["Style"], inplace=True)
    df.drop(columns=["Top Ten"], inplace=True)
    df.set_index("Review #", inplace=True)
    df[target_col] = df[target_col].astype(float)
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encode", CatBoostEncoder()),
                    ]
                ),
                categories_col,
            ),
            ("embed_input", "passthrough", [embedding_col]),
        ]
    )
    to_df = FunctionTransformer(
        func=lambda X: pd.DataFrame(X, columns=categories_col + [embedding_col]), validate=False
    )
    pipeline = Pipeline(
        [
            ("preproc", preprocessor),
            ("to_df", to_df),
            ("embed", AddEmbed(embedding_col)),
        ]
    )
    X = df[categories_col + [embedding_col]]
    y = df[[target_col]]
    X_transformed = pipeline.fit_transform(X, y)
    df_out = X_transformed.join(y.reset_index(drop=True))
    df_out.to_parquet("./data/processed/ramen_ratings_fe2.parquet")


if __name__ == "__main__":
    main()

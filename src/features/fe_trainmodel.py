import numpy as np
import pandas as pd
import pyarrow.parquet as pq

columns = [
    "trf",
    "age",
    "gndr",
    "tenure",
    "age_dev",
    "dev_man",
    "device_os_name",
    "dev_num",
    "is_dualsim",
    "is_featurephone",
    "is_smartphone",
    "simcard_type",
    "region",
    "target",
]


def main():
    df = pq.read_table("multisim_dataset.parquet", columns=columns)
    df = df.slice(0, 1000000).to_pandas()
    numeric_cols = ["age", "tenure", "age_dev", "dev_num"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df["age"] > 100, "age"] = np.nan
    df.to_parquet("./data/processed/multisim_dataset_fe3.parquet")


if __name__ == "__main__":
    main()

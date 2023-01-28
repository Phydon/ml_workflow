import pandas as pd
import numpy as np


def create_dataset(size):
    # create a fake dataset
    df = pd.DataFrame()
    df["size"] = np.random.choice(["big", "medium", "small"], size)
    df["age"] = np.random.randint(1, 50, size)
    df["team"] = np.random.choice(["red", "green", "blue", "yellow"], size)
    df["win"] = np.random.choice(["yes", "no"], size)
    dates = pd.date_range("2020-01-01", "2022-12-31")
    df["date"] = np.random.choice(dates, size)
    df["prob"] = np.random.uniform(0, 1, size)
    return df


def set_types(df):
    df["size"] = df["size"].astype("category")
    df["team"] = df["team"].astype("category")
    df["age"] = df["age"].astype("int16")
    df["win"] = df["win"].map({"yes": True, "no": False})
    df["prob"] = df["prob"].astype("float16")
    return df


def main():
    fake_df = create_dataset(1000)
    set_types(fake_df)
    print(fake_df.head())
    print(fake_df.info())
    print(fake_df.describe())

    fake_df.to_csv("fake_dataset.csv", index=False)
    fake_df.to_parquet("fake_dataset.parquet", engine="fastparquet")


if __name__ == "__main__":
    main()

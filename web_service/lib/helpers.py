import pandas as pd
from prefect import task
import pickle
from typing import List
from sklearn.feature_extraction import DictVectorizer


def load_data(path):
    return pd.read_csv(path)

@task
def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file

def save_pickle(path: str, file):
    with open(path, "wb") as f:
        pickle.dump(file, f)

def extract_x_y(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    numerical_cols: List[str] = None,
    dv: DictVectorizer = None,
    with_target: bool = True,
) -> dict:
    if categorical_cols is None:
        categorical_cols = ["NEIGHBORHOOD", "building_category", "building_class"]
    if numerical_cols is None:
        numerical_cols = ["zip_code", "total_unit", "square_feet", "house_age"]
    dicts = df[[*categorical_cols, *numerical_cols]].to_dict(orient="records")

    y = None
    if with_target:
        if dv is None:
            dv = DictVectorizer()
            dv.fit(dicts)
        y = df["price"].values

    x = dv.transform(dicts)
    return x, y, dv
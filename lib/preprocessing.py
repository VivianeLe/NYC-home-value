from fastapi import Request
import numpy as np
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import pickle

def encode_categorical_cols(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = ["zip_code", "square_feet", "year_built"]
    df[categorical_cols] = df[categorical_cols].fillna(-1).astype("int")
    df[categorical_cols] = df[categorical_cols].astype("str")
    return df

def load_preprocessor(path: str):
    with open(path, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj
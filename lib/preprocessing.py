from fastapi import Request
import numpy as np
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import pickle

def encode_cols(df: pd.DataFrame, categorical_cols: List[str] = None, numerical_cols: List[str] = None) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = ["NEIGHBORHOOD", "building_category", "building_class"]
    if numerical_cols is None:
        numerical_cols = ["zip_code", "total_unit", "square_feet", "year_built"]
    # df[categorical_cols] = df[categorical_cols].fillna(-1).astype("int")
    df[numerical_cols] = df[numerical_cols].fillna(0).astype("float")
    df[categorical_cols] = df[categorical_cols].astype("str")
    
    return df

def load_preprocessor(path: str):
    with open(path, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj
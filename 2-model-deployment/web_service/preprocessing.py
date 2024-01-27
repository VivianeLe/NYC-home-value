from fastapi import Request
import numpy as np
from pydantic import BaseModel
import pandas as pd
import pickle

def encode_cols(df: pd.DataFrame, categorical_cols: List[str] = None, numerical_cols: List[str] = None) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = ["NEIGHBORHOOD", "building_category", "building_class"]
    if numerical_cols is None:
        numerical_cols = ["zip_code", "square_feet", "year_built"]

    # Check if the specified columns exist in the DataFrame
    for col in categorical_cols + numerical_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")

    # Perform encoding and filling operations
    df[numerical_cols] = df[numerical_cols].fillna(0).astype("float")
    df[categorical_cols] = df[categorical_cols].astype("str")

    return df

def load_preprocessor(path):
    with open(path, "rb") as f:
        preprocessor = pickle.load(f)
    return preprocessor

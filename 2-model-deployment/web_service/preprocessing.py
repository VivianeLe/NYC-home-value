from typing import List
import pandas as pd
import pickle

def encode_cols(df: pd.DataFrame, numerical_cols: List[str] = None) -> pd.DataFrame:
    if numerical_cols is None:
        numerical_cols = ["zip_code", "square_feet", "year_built"]
    
    # Fill NaN values with -1 and convert to int
    df[numerical_cols] = df[numerical_cols].fillna(-1).astype("int")

    # Convert numerical columns to string
    df[numerical_cols] = df[numerical_cols].astype("str")
    
    return df

def load_preprocessor(path):
    with open(path, "rb") as f:
        preprocessor = pickle.load(f)
    return preprocessor

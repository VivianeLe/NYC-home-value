from typing import List
import pandas as pd
from config import CATEGORICAL_COLS, NUMERICAL_COLS

def encode_cols(df: pd.DataFrame, categorical_cols: List[str] = None, numerical_cols: List[str] = None) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS

    df[numerical_cols] = df[numerical_cols].fillna(-1).astype("str")
    df[categorical_cols] = df[categorical_cols].astype("str")
    return df
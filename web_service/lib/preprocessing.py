from typing import List
import pandas as pd
from config import CATEGORICAL_COLS, NUMERICAL_COLS, PATH_TO_PREPROCESSOR, PATH_TO_TRAIN, PATH_TO_TEST
from prefect import task
from lib.helpers import load_data, extract_x_y, save_pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from datetime import datetime

def encode_cols(df: pd.DataFrame, categorical_cols: List[str] = None, numerical_cols: List[str] = None) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS

    df[numerical_cols] = df[numerical_cols].fillna(-1).astype("float")
    df[categorical_cols] = df[categorical_cols].apply(lambda x: x.astype(str).str.lower())
    return df

def clean_data(df):
    df['SALE DATE'] = pd.to_datetime(df['SALE DATE'], format="%d-%m-%Y %H:%M")
    df['house_age'] = df['SALE DATE'].dt.year - df['YEAR BUILT']
    df = df[(df['ZIP CODE']!=0) & (df['YEAR BUILT']!=0)]
    df = df[['NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'BUILDING CLASS AT PRESENT', 'ZIP CODE','TOTAL UNITS', 'GROSS SQUARE FEET', 'house_age', 'SALE PRICE']]
    df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'], errors='coerce')
    df['GROSS SQUARE FEET'] = pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
    df.rename(columns={'BUILDING CLASS CATEGORY': 'building_category',
                    'BUILDING CLASS AT PRESENT': 'building_class',
                    'ZIP CODE': 'zip_code',
                    'TOTAL UNITS': 'total_unit',
                    'GROSS SQUARE FEET': 'square_feet',
                    'SALE PRICE': 'price'
                    }, inplace=True)
    df['building_category'] = df['building_category'].str[:3].apply(lambda x: x.strip())
    columns_to_replace = ['total_unit', 'square_feet', 'house_age']
    mean_values = df[columns_to_replace].mean()
    df[columns_to_replace] = df[columns_to_replace].replace(0, mean_values)
    df[columns_to_replace] = df[columns_to_replace].fillna(mean_values)
    df = df.dropna()
    df = df[(df['price'] >=20000) & (df['price'] <=3000000)]
    return df

@task(name="Load, clean, split data")
def load_clean_split(data_path):
    df = load_data(data_path)
    df = clean_data(df)
    df.to_csv("/Users/viviane/Desktop/MLOps/NYC-home-value/data/nyc-house-price-cleaned.csv")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(PATH_TO_TRAIN)
    test_df.to_csv(PATH_TO_TEST)
    return train_df, test_df

@task(name="Preprocess data")
def run_encode_task(df, dv: DictVectorizer = None) -> np.ndarray:
    df = encode_cols(df)
    x, y, dv = extract_x_y(df, CATEGORICAL_COLS, NUMERICAL_COLS, dv)
    return x, y, dv
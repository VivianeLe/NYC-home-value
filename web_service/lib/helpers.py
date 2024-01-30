import pandas as pd
from datetime import datetime
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

@task
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
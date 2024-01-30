
from lib.preprocessing import encode_cols
from lib.helpers import extract_x_y, load_data, clean_data
from lib.modeling import  predict_price, train_model, evaluate_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from config import CATEGORICAL_COLS, NUMERICAL_COLS
from prefect import flow, task
import logging

logger = logging.getLogger(__name__)

@flow(name="Preprocess data")
def run_encode_flow(df, dv: DictVectorizer = None) -> np.ndarray:
    logger.info(f"Encoded DataFrame: {df}")
    df = encode_cols(df)
    x, y, dv = extract_x_y(df, CATEGORICAL_COLS, NUMERICAL_COLS, dv)
    logger.info(f"Extracted x, y: {x}, {y}")
    return x, y, dv

@flow(name="Train model", retries=3, retry_delay_seconds=30, log_prints=True)
def train_model_flow(train_df: pd.DataFrame = None, model_type: str = None):
    x_train, y_train, dv = run_encode_flow(train_df)
    try:
        model = train_model(x_train=x_train, y_train=y_train, model_type=model_type)
    except ValueError as e:
        logger.error(f"An error occurred while training the model: {e}")
        raise
    logger.info(f"Successfully trained the model: {model_type}")
    return model, dv

@flow(name="Predict and evaluate", retries=3, retry_delay_seconds=10, log_prints=True)
def predict_model_flow(test_df: pd.DataFrame = None, model: BaseEstimator = None, dv: DictVectorizer = None):
    x_test, y_test, _ = run_encode_flow(test_df, dv)
    if model is None:
        raise ValueError("Input model cannot be None.")
    try: 
        pred = predict_price(input_data=x_test, model=model)
    except ValueError as e:
        logger.error(f"An error occurred while predicting the value: {e}")
        raise
    rmse, mae, r2 =  evaluate_model(y_true=y_test, y_pred=pred)
    logger.info(f"Model's evaluation: {rmse, mae, r2}")
    return rmse, mae, r2

if __name__ == "__main__":
    df = load_data('/Users/viviane/Desktop/MLOps/NYC-home-value/data/nyc-rolling-sales.csv')
    df = clean_data(df)
    df.to_csv("/Users/viviane/Desktop/MLOps/NYC-home-value/data/nyc-house-price-cleaned.csv")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    model, dv = train_model_flow(train_df, model_type="xgb")
    rmse, mae, r2 = predict_model_flow(test_df, model, dv)
    print(f"Model evaluation: RMSE: {rmse} | MAE: {mae} | R2: {r2}")

from lib.preprocessing import run_encode_task, load_clean_split
from lib.helpers import save_pickle
from lib.modeling import  predict_price, train_model, evaluate_model
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from config import PATH_TO_MODEL
from prefect import flow, serve, task
import logging

logger = logging.getLogger(__name__)

@flow(name="Train model", retries=3, retry_delay_seconds=30, log_prints=True)
def train_model_flow(train_df: pd.DataFrame = None, model_type: str = None):
    x_train, y_train, dv = run_encode_task(train_df)
    try:
        model = train_model(x_train=x_train, y_train=y_train, model_type=model_type)
        save_pickle(PATH_TO_MODEL, model)
    except ValueError as e:
        logger.error(f"An error occurred while training the model: {e}")
        raise
    logger.info(f"Successfully trained the model: {model_type}")
    return model, dv

@flow(name="Predict and evaluate", retries=3, retry_delay_seconds=10, log_prints=True)
def predict_model_flow(test_df: pd.DataFrame = None, model: BaseEstimator = None, dv: DictVectorizer = None):
    x_test, y_test, _ = run_encode_task(test_df, dv)
    if model is None:
        raise ValueError("Input model cannot be None.")
    try: 
        pred = predict_price(input_data=x_test, model=model)
        rmse, mae, r2 =  evaluate_model(y_true=y_test, y_pred=pred)
    except ValueError as e:
        logger.error(f"An error occurred while predicting the value: {e}")
        raise
    logger.info(f"Model's evaluation: {rmse, mae, r2}")
    return rmse, mae, r2

@flow(name="NYC house price predict - main flow", log_prints=True)
def main_flow(data_path, model_type):
    train_df, test_df = load_clean_split(data_path)
    model, dv = train_model_flow(train_df, model_type)
    rmse, mae, r2 = predict_model_flow(test_df, model, dv)
    evaluation = f"Model evaluation: RMSE: {rmse} | MAE: {mae} | R2: {r2}"
    return evaluation

if __name__ == "__main__":
    data_path = '/Users/viviane/Desktop/MLOps/NYC-home-value/data/nyc-rolling-sales.csv'
    model_type = 'xgb'
    deploy = main_flow.to_deployment(
        name='NYC House Price Deployment',
        version='0.1.0',
        tags=['house predict'],
        interval=600,
        parameters={
            'data_path': data_path,
            'model_type': model_type
        }
    )
    serve(deploy)

#uvicorn main:app --reload
from prefect import task, flow
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@task
def train_model(x_train: pd.DataFrame, y_train: np.ndarray, model_type):
    random = RandomForestRegressor(random_state=42, n_estimators=25,
                              max_depth=60, min_samples_leaf=1, min_samples_split=5)
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.25, n_estimators=300,
                                max_depth=6, subsample=1, colsample_bytree=1)
    linear = LinearRegression(fit_intercept=True, copy_X=True)

    model_map = {
        "randomforest": random,
        "xgb": xgb_reg,
        "linear": linear
    }
    model = model_map.get(model_type, None)
    if model is None:
        raise ValueError(f"Invalid model type: {model_type}")
    
    model.fit(x_train, y_train)
    return model

@task
def predict_price(input_data, model):
    return model.predict(input_data)

@task
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
    mae = round(mean_absolute_error(y_true, y_pred), 2)
    r2 = round(r2_score(y_true, y_pred), 4)
    return rmse, mae, r2

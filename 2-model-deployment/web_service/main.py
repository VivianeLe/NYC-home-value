from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from preprocessing import encode_cols, load_preprocessor
from lib.modeling import get_model
from sklearn.feature_extraction import DictVectorizer
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from config import PATH_TO_PREPROCESSOR, PATH_TO_MODEL, CATEGORICAL_COLS, NUMERICAL_COLS
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

class InputData(BaseModel):
    zip_code: int
    square_feet: int
    year_built: int

config = {}
# Load configuration from config.json
try:
    with open("config.py", "r") as config_file:
        exec(config_file.read(), config)
except Exception as e:
    logger.error(f"Error loading configuration: {e}")

def run_inference(user_input: List[InputData], dv: DictVectorizer, model: BaseEstimator) -> np.ndarray:
    try:
        df = pd.DataFrame([x.dict() for x in user_input])
        df = encode_cols(df)
        dicts = df[NUMERICAL_COLS].to_dict(orient="records")
        X = dv.transform(dicts)
        y = model.predict(X)
        return y
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail="Error during inference")

@app.get("/")
def read_root():
    return {"message": "Welcome to the NYC Home value prediction!"}

@app.post("/predict_duration")
def predict_duration_route(payload: InputData):
    try:
        dv = load_preprocessor(PATH_TO_PREPROCESSOR)
        model = get_model(PATH_TO_MODEL)
        y = run_inference([payload], dv, model)
        return {"House_price_value: ": y[0]}
    except Exception as e:
        logger.error(f"Error predicting duration: {e}")
        raise HTTPException(status_code=500, detail="Error predicting duration")

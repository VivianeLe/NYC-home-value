from fastapi import FastAPI
from pydantic import BaseModel
from lib.preprocessing import encode_cols
from lib.helpers import load_pickle
from sklearn.feature_extraction import DictVectorizer
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from config import PATH_TO_PREPROCESSOR, PATH_TO_MODEL
import logging
from prefect import flow

logger = logging.getLogger(__name__)

app = FastAPI()
class InputData(BaseModel):
    NEIGHBORHOOD: str
    building_category: str
    building_class: str
    zip_code: int
    total_unit: int
    square_feet: float
    house_age: float

@flow()
def run_inference(user_input: List[InputData], dv: DictVectorizer, model: BaseEstimator) -> np.ndarray:
    df = pd.DataFrame([x.dict() for x in user_input])
    df = encode_cols(df)
    dicts = df.to_dict(orient="records")
    X = dv.transform(dicts)
    y = model.predict(X)
    logger.info(f"Predicted house price: {y} USD")
    return y

@app.get("/")
def read_root():
    return {"message": "This is NYC House Price Prediction App"}

@app.post("/predict_house_price")
@flow(name="Single prediction")
def predict_house_price(payload: InputData):
    dv = load_pickle(PATH_TO_PREPROCESSOR)
    model = load_pickle(PATH_TO_MODEL)
    y = run_inference([payload], dv, model)
    result = f"Predicted house price: {round(y[0]):,.0f} USD"
    return {result}

# Run FastAPI
#uvicorn main:app --reload
#check MLflow
# mlflow ui --host 0.0.0.0 --port 5002
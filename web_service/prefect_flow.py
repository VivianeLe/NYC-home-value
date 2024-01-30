
from lib.preprocessing import run_encode_task, load_clean_split
from lib.helpers import load_data, save_pickle, load_pickle
from lib.modeling import  predict_price, train_model, evaluate_model
from config import PATH_TO_MODEL, PATH_TO_PREPROCESSOR, PATH_TO_TEST, PATH_TO_TRAIN
from prefect import flow
import logging
import mlflow.sklearn

logger = logging.getLogger(__name__)

@flow(name="Train model", retries=3, retry_delay_seconds=30, log_prints=True)
def train_model_flow(model_type: str = None):
    df = load_data(PATH_TO_TEST)
    x_train, y_train, dv = run_encode_task(df)
    try:
        model = train_model(x_train=x_train, y_train=y_train, model_type=model_type)
        save_pickle(PATH_TO_MODEL, model)
        logger.info('Model trained successfully')
    except ValueError as e:
        logger.error(f"An error occurred while training the model: {e}")
        raise
    return model

@flow(name="Predict and evaluate", retries=3, retry_delay_seconds=10, log_prints=True)
def predict_model_flow():
    df = load_data(PATH_TO_TRAIN)
    model = load_pickle(PATH_TO_MODEL)
    dv = load_pickle(PATH_TO_PREPROCESSOR)
    x_test, y_test, _ = run_encode_task(df, dv)
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

@flow(name="Train and predict - main flow", log_prints=True)
def main_flow(model_type):
    mlflow_experiment_path = f"nyc_house_price_{model_type}"
    mlflow.set_experiment(mlflow_experiment_path)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("stage", "training")

        model = train_model_flow(model_type)
        mlflow.sklearn.log_model(model, "model")
        params = model.get_params()
        mlflow.log_params(params)

        rmse, mae, r2 = predict_model_flow()
        mlflow.log_metric("Test-MSE", rmse)
        mlflow.log_metric("Test-MAE", mae)
        mlflow.log_metric("Test-R2", r2)
        evaluation = f"Model evaluation: RMSE: {rmse} | MAE: {mae} | R2: {r2}"
        mlflow.register_model(f"runs:/{run_id}/model", mlflow_experiment_path)
    return evaluation

if __name__ == "__main__":
    model_type = 'randomforest'
    main_flow.serve(
        name='NYC House Price Full Deployment',
        version='0.1.0',
        tags=['train', 'predict'],
        interval=604800,
        parameters={
            'model_type': model_type
        }
    )
    predict_model_flow.serve(name="NYC-house-predict",
                        tags=['predict'],
                        interval=600
                        )

#uvicorn main:app --reload
# mlflow ui --host 0.0.0.0 --port 5002
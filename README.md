# NYC house price prediction
This project is aim to:
- build a model to predict house pricing in NYC
- log the model's performances and save it to the MLflow registry
- run with Fast API to allow user input values to predict price
- build image on Docker and run image

The raw dataset contains 84548 samples, 22 columns. We only select some useful columns to train model:
['NEIGHBORHOOD', 'BUILDING CLASS AT PRESENT', 'ZIP CODE', 'TOTAL UNITS', 'GROSS SQUARE FEET', 'YEAR BUILT', 'SALE PRICE']
After removing NA values, the new dataset includes 48248 samples.

I/ Required application:
GIT
Docker
VSCode

II/ Detail steps
1. Clean data
- Remove NA values
- Calculate house age
- With columns 'total_unit', 'square_feet', 'house_age', replace 0 and NA value with mean value.
- Filter only the sample with price from 20k - 3M

After cleaning, here is description of dataset: (50443 sample, 8 columns)

<img width="560" alt="Screen Shot 2024-01-31 at 18 39 18" src="https://github.com/VivianeLe/NYC-home-value/assets/95589311/89e8808c-8da9-44c8-a9dc-5c31b256bb11">

2. Data visualization
Build dashboard on Looker:
Link to dashboard: https://shorturl.at/abwLT

3. Build functions

4. Run MLFlow
From your terminal, run: 
mlflow ui --host 0.0.0.0 --port 5002

5. Run FastAPI:
From your terminal, go to the directory that contain main.py file, run:
unicorn main:app —reload

6. Run Docker:
Build an image name house-predict on docker
From your terminal, run:
docker run -p 0.0.0.0:8000:8001 house-predict

7. Run Prefect:
Asign @task and @flow for the functions. There are 2 flows in main_flow: train_flow and predict_flow
From your terminal, go to the directory that contains prefect_flow.py file, run:
python -i prefect_flow.py

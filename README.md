# NYC house price prediction
This project is aim to:
- build a model to predict house pricing in NYC
- log the model's performances and save it to the MLflow registry
- run with Fast API to allow user input values to predict price
- build image on Docker and run image

**I/ Required application:**

GIT
Docker
VSCode

**II/ Detail steps**

_1. Clean data_

The raw dataset contains 84548 samples, 22 columns. We only select some useful columns to train model
- Remove NA values
- Calculate house age
- With columns 'total_unit', 'square_feet', 'house_age', replace 0 and NA value with mean value.
- Filter only the sample with price from 20k - 3M

After cleaning, here is description of dataset: (50443 sample, 8 columns)

<img width="560" alt="Screen Shot 2024-01-31 at 18 39 18" src="https://github.com/VivianeLe/NYC-home-value/assets/95589311/89e8808c-8da9-44c8-a9dc-5c31b256bb11">

_2. Data visualization_

Build dashboard on Looker:

Link to dashboard: https://shorturl.at/abwLT

<img width="636" alt="Screen Shot 2024-01-31 at 18 57 18" src="https://github.com/VivianeLe/NYC-home-value/assets/95589311/ddc47da0-fd41-4f7d-b56a-7267ce1271dd">

_3. Build functions_

Select features to be used in training model by using Generalized Linear Model Regression

X = ['NEIGHBORHOOD', 'BUILDING CLASS AT PRESENT', 'ZIP CODE', 'TOTAL UNITS', 'GROSS SQUARE FEET', 'YEAR BUILT']

Y = 'SALE PRICE'

_4. Run MLFlow_

From your terminal, run: 

mlflow ui --host 0.0.0.0 --port 5002

<img width="1381" alt="Screen Shot 2024-01-31 at 18 58 11" src="https://github.com/VivianeLe/NYC-home-value/assets/95589311/75ddffae-137d-43c4-aed3-d4474e1ac669">

_5. Run Docker_

Build an image name house-predict on docker

From your terminal, run:

docker run -p 0.0.0.0:8000:8001 house-predict
<img width="1018" alt="Screen Shot 2024-01-31 at 18 58 58" src="https://github.com/VivianeLe/NYC-home-value/assets/95589311/16f07c17-210b-4ffe-b800-accb684aa760">
<img width="993" alt="Screen Shot 2024-01-31 at 18 59 09" src="https://github.com/VivianeLe/NYC-home-value/assets/95589311/0d276e42-dd04-49d1-a6bc-38d627eace3d">

_6. Run FastAPI_

From your terminal, go to the directory that contain main.py file, run:
unicorn main:app —reload

<img width="427" alt="Screen Shot 2024-01-31 at 19 00 10" src="https://github.com/VivianeLe/NYC-home-value/assets/95589311/97bbc6a8-bb93-4610-b9e7-9afd1be02cf3">
<img width="518" alt="Screen Shot 2024-01-31 at 19 00 42" src="https://github.com/VivianeLe/NYC-home-value/assets/95589311/129df856-a2d8-44f0-92d3-bb928bd8814c">

_7. Run Prefect_

Asign @task and @flow for the functions. There are 2 flows in main_flow: train_flow and predict_flow

From your terminal, go to the directory that contains prefect_flow.py file, run:

python -i prefect_flow.py

Schedule the deployment: 
<img width="1376" alt="Screen Shot 2024-01-31 at 19 01 28" src="https://github.com/VivianeLe/NYC-home-value/assets/95589311/a92828b6-af2d-44fc-b0e3-8691111ee83a">
Allow user to input model type and path to dataset:
<img width="1087" alt="Screen Shot 2024-01-31 at 19 05 05" src="https://github.com/VivianeLe/NYC-home-value/assets/95589311/2c3a8e2c-a20d-409c-8b51-d1cc54a86266">
Flow runs:
<img width="1003" alt="Screen Shot 2024-01-31 at 19 02 09" src="https://github.com/VivianeLe/NYC-home-value/assets/95589311/f38ac39a-e163-44e6-8e96-7a8d1c781c7a">
Flow runs pipeline:
<img width="902" alt="Screen Shot 2024-01-31 at 19 03 22" src="https://github.com/VivianeLe/NYC-home-value/assets/95589311/8c076a71-2e0a-4c09-a2f0-97b6188be0ed">





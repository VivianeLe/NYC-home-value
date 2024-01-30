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

zip_code	total_unit	square_feet	house_age	price
count	50443.000000	50443.000000	5.044300e+04	50443.000000	5.044300e+04
mean	10865.439546	2.057191	3.504700e+03	65.823786	7.429574e+05
std	561.491431	14.041635	1.855024e+04	33.235075	5.515599e+05
min	10001.000000	1.000000	1.200000e+02	1.000000	2.000000e+04
25%	10306.000000	1.000000	1.746000e+03	47.000000	3.701100e+05
50%	11211.000000	2.000000	4.032000e+03	67.000000	5.975000e+05
75%	11361.500000	2.394866	4.256918e+03	92.000000	9.280000e+05
max	11694.000000	2261.000000	3.750565e+06	217.000000	3.000000e+06

2. Build functions

3. Run MLFlow
From your terminal, run: 
mlflow ui --host 0.0.0.0 --port 5002

3. Run FastAPI:
From your terminal, go to the directory that contain main.py file, run:
unicorn main:app —reload

4. Run Docker:
Build an image name house-predict on docker
From your terminal, run:
docker run -p 0.0.0.0:8000:8001 house-predict

5. Run Prefect:
Asign @task and @flow for the functions. There are 2 flows in main_flow: train_flow and predict_flow
From your terminal, go to the directory that contains prefect_flow.py file, run:
python -i prefect_flow.py
# NYC house price prediction
This project is aim to:
- build a model to predict house pricing in NYC
- log the model's performances and save it to the MLflow registry
- run with Fast API to allow user input values to predict price
- build image on Docker and run image

The raw dataset contains 84548 samples, 22 columns. We only select some useful columns to train model:
['NEIGHBORHOOD', 'BUILDING CLASS AT PRESENT', 'ZIP CODE', 'TOTAL UNITS', 'GROSS SQUARE FEET', 'YEAR BUILT', 'SALE PRICE']
After removing NA values, the new dataset includes 48248 samples.

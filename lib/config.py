# MODELS
MODEL_VERSION = "0.0.1"
MODEL_TYPE = "ensemble"
PATH_TO_PREPROCESSOR = f"/Users/viviane/Desktop/MLOps/NYC-home-value/saved_pkl/dv/{MODEL_TYPE}-{MODEL_VERSION}.pkl"
PATH_TO_MODEL = f"/Users/viviane/Desktop/MLOps/NYC-home-value/saved_pkl/model/{MODEL_TYPE}-{MODEL_VERSION}.pkl"
CATEGORICAL_COLS = ["NEIGHBORHOOD", "building_category", "building_class"]
NUMERICAL_COLS = ["zip_code", "total_unit", "square_feet", "year_built"]

# MISC
APP_TITLE = "NYC House Price Prediction App"
APP_DESCRIPTION = (
    "A simple API to predict house pricing in minutes "
    "for NYC house, given a neighborhood, a building category, building class, zip code "
    "total unit, square feet and year built."
)
APP_VERSION = "0.0.1"

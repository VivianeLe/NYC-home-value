from sklearn.feature_extraction import DictVectorizer
import pickle
from app_config import PATH_TO_PREPROCESSOR

# preprocess_path = "C:\Git\Project-1\NYC-home-value\saved_pkl\dv\0.0.1.pkl"

def save_picked(path: str, dv: DictVectorizer):
    with open(path, "wb") as f:
        pickle.dump(dv, f)

dv = DictVectorizer()
save_picked(PATH_TO_PREPROCESSOR, dv)
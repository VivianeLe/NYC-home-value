from pickle import load

def load_pickle(model_path: str):
    with open(model_path, 'rb') as f:
        pkl_file = load(f)
    return pkl_file
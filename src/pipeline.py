from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from preprocess import create_preprocessor

def build_pipeline(X):

    preprocessor = create_preprocessor(X)

    model = RandomForestRegressor()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline
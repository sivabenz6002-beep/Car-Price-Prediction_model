import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, f_regression


def remove_outliers(df):

    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns

    for col in numeric_cols:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


def encode_target(y):

    if y.dtype == "object":

        le = LabelEncoder()
        y = le.fit_transform(y)

    return y


def create_preprocessor(X):

    numeric_features = X.select_dtypes(
        include=["int64","float64"]
    ).columns

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns

    # Numeric preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor


def feature_selection():

    selector = SelectKBest(
        score_func=f_regression,
        k="all"
    )

    return selector
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from pipeline import build_pipeline


# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/car_data.csv")

# remove index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("Dataset Columns:")
print(df.columns)

# ---------------- TARGET ----------------
target = "selling_price"

X = df.drop(columns=[target])
y = df[target]


# ---------------- PIPELINE ----------------
pipeline = build_pipeline(X)


# ---------------- HYPERPARAMETERS ----------------
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None],
    "model__min_samples_split": [2, 5]
}


# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------- GRID SEARCH ----------------
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("\nTraining model...\n")

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:")
print(grid.best_params_)


# ---------------- PREDICTIONS ----------------
y_pred = best_model.predict(X_test)


# ---------------- METRICS ----------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5


print("\nModel Performance")
print("------------------")
print("R2 Score :", r2)
print("MAE      :", mae)
print("MSE      :", mse)
print("RMSE     :", rmse)


# ---------------- SAVE MODEL ----------------
joblib.dump(best_model, "models/car_price_model.pkl")

print("\nModel saved successfully at:")
print("models/car_price_model.pkl")
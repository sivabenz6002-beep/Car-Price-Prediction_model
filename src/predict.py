import joblib
import pandas as pd

model = joblib.load("models/car_price_model.pkl")

def predict_price(input_data):

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)

    return prediction[0]
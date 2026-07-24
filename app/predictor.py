import joblib
import pandas as pd


class Predictor:

    def __init__(self):
        self.model = joblib.load("artifacts/model.pkl")

    def predict(self, data):

        df = pd.DataFrame([data])

        prediction = self.model.predict(df)

        return float(prediction[0])
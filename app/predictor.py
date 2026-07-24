import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "model.pkl"


class Predictor:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, data):
        df = pd.DataFrame([data])
        prediction = self.model.predict(df)
        return float(prediction[0])
from fastapi import FastAPI
from app.predictor import Predictor
from app.schemas import CarFeatures

app = FastAPI(
    title="Car Price Prediction API",
    version="1.0.0"
)

predictor = Predictor()


@app.get("/")
def home():

    return {
        "message": "Car Price Prediction API is Running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(car: CarFeatures):

    prediction = predictor.predict(car.model_dump())

    return {
        "predicted_price": round(prediction, 2)
    }
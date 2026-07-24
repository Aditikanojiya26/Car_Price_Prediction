import dagshub
import mlflow


def setup_mlflow():

    dagshub.init(
        repo_owner="Aditikanojiya26",
        repo_name="car-price-prediction",
        mlflow=True
    )

    mlflow.set_experiment(
        "Car Price Prediction"
    )

    
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import pandas as pd


class ModelTraining:

    def split_data(self, df):

        X = df.drop(
            columns=[
                "price_number",
                "car_prices_in_rupee"
            ],
            errors="ignore"
        )

        y = df["price_number"]

        print(X.columns)

        y_bucket = pd.qcut(
            y,
            q=10,
            labels=False,
            duplicates="drop"
        )

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y_bucket
        )

    def get_pipeline(self, preprocessor):

        pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            tree_method="hist",
            device="cuda",

            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,

            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=1.0,

            min_child_weight=3,
            gamma=0,

            reg_alpha=0.1,
            reg_lambda=10
        ))
    ])

        return pipeline
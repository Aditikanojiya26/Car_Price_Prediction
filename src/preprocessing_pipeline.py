from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OneHotEncoder
)

from category_encoders import (
    BinaryEncoder,
    TargetEncoder
)

import numpy as np


class PreprocessingPipeline:

    def get_preprocessor(self):

       

        transmission_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        fuel_type_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        brand_target_pipeline = Pipeline([
            ("target", TargetEncoder())
        ])

        car_name_binary_pipeline = Pipeline([
            ("binary", BinaryEncoder())
        ])

        preprocessor_target = ColumnTransformer(
            transformers=[
                (
                    "transmission",
                    transmission_pipeline,
                    ["transmission"]
                ),
                (
                    "fuel_type",
                    fuel_type_pipeline,
                    ["fuel_type"]
                ),
                (
                    "brand",
                    brand_target_pipeline,
                    ["brand"]
                ),
                (
                    "car_name",
                    car_name_binary_pipeline,
                    ["car_name"]
                )
            ],
            remainder="passthrough"
        )

        return preprocessor_target
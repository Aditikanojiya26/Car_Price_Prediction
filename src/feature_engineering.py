import pandas as pd


class FeatureEngineering:

    def transform(self, df):

        # Drop unnecessary columns
        cols_to_drop = [
            "Unnamed: 0",
            "car_prices_in_rupee"
        ]

        existing_cols = [
            col for col in cols_to_drop
            if col in df.columns
        ]

        df = df.drop(columns=existing_cols)

        print("Feature Engineering Completed")

        return df
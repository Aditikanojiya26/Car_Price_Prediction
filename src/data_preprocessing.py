import pandas as pd
import numpy as np


class DataPreprocessing:

    def __init__(self):
        pass

    def clean_price(self,value):

        value = str(value).strip()

        if "Lakh" in value:
            value = value.replace("Lakh", "").replace(",", "").strip()
            return int(float(value) * 100000)

        elif "Crore" in value:
            value = value.replace("Crore", "").replace(",", "").strip()
            return int(float(value) * 10000000)

        else:
            value = value.replace(",", "").strip()
            return int(float(value))

    def clean_kms_column(self, col):

        return (
            col.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(r"\D+", "", regex=True)
            .astype(int)
        )

    def clean_owner(self, value):

        value = str(value)

        return int(value.split()[0][0])


    def clean_engine(self, value):

        return int(str(value).split()[0])

    def extract_brand(self, car_name):

        if pd.isna(car_name):
            return pd.NA

        car_name = str(car_name).strip()

        if car_name.startswith("Land Rover"):
            return "Land Rover"

        return car_name.split()[0]
    def clean_seats(self, value):
        return int(str(value).split()[0])

    def preprocess(self, df):

        print("Starting preprocessing...")

        # cleaning logic here
        df["car_age"] = 2022 - df["manufacture"]
        df["price_number"] = (df["car_prices_in_rupee"].apply(self.clean_price))
        df["kms_driven"] = self.clean_kms_column(df["kms_driven"])
        df["ownership"] = (df["ownership"].apply(self.clean_owner))
        df["engine"] = (df["engine"].apply(self.clean_engine))
        df["brand"] = (df["car_name"].apply(self.extract_brand))
        df["Seats"] = (df["Seats"].apply(self.clean_seats))



        print("Preprocessing completed")

        return df
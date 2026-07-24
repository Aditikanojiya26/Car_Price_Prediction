import pandas as pd
from pathlib import Path


class DataIngestion:

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        """

        file_path = Path(self.data_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {self.data_path}"
            )

        df = pd.read_csv(file_path)

        print("=" * 50)
        print("DATA INGESTION COMPLETED")
        print("=" * 50)
        print(f"Dataset Shape : {df.shape}")
        print(f"Rows          : {df.shape[0]}")
        print(f"Columns       : {df.shape[1]}")
        print("=" * 50)

        return df
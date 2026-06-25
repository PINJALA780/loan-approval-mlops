import os
import pandas as pd

class DataIngestion:

    def __init__(self):
        self.raw_data_path = "data/raw/loan_data.csv"

    def initiate_data_ingestion(self):

        print("Reading Dataset")

        df = pd.read_csv(self.raw_data_path)

        print(df.head())

        return self.raw_data_path


if __name__ == "__main__":

    obj = DataIngestion()

    obj.initiate_data_ingestion()

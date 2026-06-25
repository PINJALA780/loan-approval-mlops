import pandas as pd

class DataValidation:

    def validate_dataset(self):

        df = pd.read_csv("data/raw/loan_data.csv")

        print("Dataset Shape")

        print(df.shape)

        print("\nMissing Values\n")

        print(df.isnull().sum())


if __name__ == "__main__":

    obj = DataValidation()

    obj.validate_dataset()

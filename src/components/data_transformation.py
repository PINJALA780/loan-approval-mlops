import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataTransformation:

    def transform_data(self):

        df = pd.read_csv("data/raw/loan_data.csv")

        # Remove extra spaces from column names
        df.columns = df.columns.str.strip()

        df.fillna(method="ffill", inplace=True)

        encoder = LabelEncoder()

        for col in df.select_dtypes(include="object").columns:
            df[col] = encoder.fit_transform(df[col])

        df.to_csv(
            "data/processed/processed.csv",
            index=False
        )

        print("Transformation Complete")


if __name__ == "__main__":

    DataTransformation().transform_data()

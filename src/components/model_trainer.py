import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class ModelTrainer:

    def train_model(self):

        mlflow.set_experiment("loan_approval_prediction")

        df = pd.read_csv("data/processed/processed.csv")

        # Remove extra spaces
        df.columns = df.columns.str.strip()

        print("\nColumns in dataset:")
        print(df.columns.tolist())

        target_col = "loan_status"

        if target_col not in df.columns:
            raise Exception(
                f"loan_status column not found.\nAvailable columns: {df.columns.tolist()}"
            )

        # Remove ID column if present
        if "loan_id" in df.columns:
            df = df.drop(columns=["loan_id"])

        X = df.drop(columns=[target_col])

        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        with mlflow.start_run():

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, pred)

            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("test_size", 0.2)

            mlflow.log_metric("accuracy", accuracy)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="loan_approval_model"
            )

            print(f"\nAccuracy: {accuracy:.4f}")

            joblib.dump(
                model,
                "models/model.pkl"
            )

            print("\nModel Saved Successfully")
            print("MLflow Run Logged Successfully")


if __name__ == "__main__":
    obj = ModelTrainer()
    obj.train_model()

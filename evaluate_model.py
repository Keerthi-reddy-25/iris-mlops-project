
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score


def main():
    try:
        data = pd.read_csv("iris_dataset.csv")
    except FileNotFoundError:
        iris = load_iris(as_frame=True)
        data = iris.frame.rename(columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
            "target": "species",
        })
        data["species"] = data["species"].map(
            dict(enumerate(iris.target_names))
        )

    features = [
        "sepal_length", "sepal_width",
        "petal_length", "petal_width"
    ]
    data = data.dropna(subset=features + ["species"]).drop_duplicates()

    X = data[features]
    encoder = LabelEncoder()
    y = encoder.fit_transform(data["species"].astype(str))

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    bundle = joblib.load("iris_model.pkl")
    model = bundle["model"]
    scaler = bundle["scaler"]
    saved_encoder = bundle["encoder"]

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2%}")

    if list(encoder.classes_) != list(saved_encoder.classes_):
        raise SystemExit("ERROR: Model class labels do not match dataset.")

    if accuracy < 0.90:
        raise SystemExit("ERROR: Test accuracy is below 90%.")

    print("PASS: Model accuracy meets the 90% threshold.")


if __name__ == "__main__":
    main()

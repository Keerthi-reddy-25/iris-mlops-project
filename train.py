import pandas as pd
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load CSV, or use the built-in Iris dataset if the CSV is missing.
try:
    data = pd.read_csv("iris_dataset.csv")
except FileNotFoundError:
    iris = load_iris(as_frame=True)
    data = iris.frame.rename(columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "target": "species"
    })
    data["species"] = data["species"].map(dict(enumerate(iris.target_names)))

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
data = data.dropna(subset=features + ["species"]).drop_duplicates()
X = data[features]
encoder = LabelEncoder()
y = encoder.fit_transform(data["species"].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=1,
    random_state=42
)

EPOCHS = 100
for epoch in range(EPOCHS):
    model.partial_fit(X_train, y_train, classes=list(range(len(encoder.classes_))))
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    print(f"Epoch {epoch + 1}/{EPOCHS} - Training Accuracy: {train_accuracy:.4f}")

y_pred = model.predict(X_test)
print("\nDataset rows:", len(data))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=list(range(len(encoder.classes_))),
    target_names=encoder.classes_,
    zero_division=0
))

joblib.dump({
    "model": model,
    "scaler": scaler,
    "encoder": encoder,
    "features": features
}, "iris_model.pkl")
print("\nModel and preprocessing objects saved to iris_model.pkl")

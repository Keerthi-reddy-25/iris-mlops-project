import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
old_data = pd.read_csv("iris_dataset.csv")
new_data = pd.read_csv("new_iris_data.csv")

if "species" not in new_data.columns or new_data["species"].isna().any():
    raise ValueError("New dataset must include correct, non-empty species labels.")

combined = pd.concat([old_data, new_data], ignore_index=True)
combined = combined.dropna(subset=features + ["species"]).drop_duplicates()
encoder = LabelEncoder()
y = encoder.fit_transform(combined["species"].astype(str))
X = combined[features]

if pd.Series(y).value_counts().min() < 2:
    raise ValueError("Each species needs at least two rows for stratified splitting.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
    learning_rate_init=0.01, max_iter=1, random_state=42
)

EPOCHS = 100
for epoch in range(EPOCHS):
    model.partial_fit(X_train, y_train, classes=list(range(len(encoder.classes_))))
    print(f"Epoch {epoch + 1}/{EPOCHS} - Training Accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")

predictions = model.predict(X_test)
print(f"\nUpdated model test accuracy: {accuracy_score(y_test, predictions):.2%}")
print(classification_report(
    y_test, predictions, labels=list(range(len(encoder.classes_))),
    target_names=encoder.classes_, zero_division=0
))
joblib.dump({
    "model": model, "scaler": scaler, "encoder": encoder, "features": features
}, "iris_model_updated.pkl")
print("Updated model saved as iris_model_updated.pkl")

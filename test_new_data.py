import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

bundle = joblib.load("iris_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
encoder = bundle["encoder"]
features = bundle["features"]

data = pd.read_csv("new_iris_data.csv")
missing = [col for col in features if col not in data.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

X_new = data[features]
predicted_ids = model.predict(scaler.transform(X_new))
data["predicted_species"] = encoder.inverse_transform(predicted_ids)

print("Predictions on new data:")
print(data.to_string(index=False))

if "species" in data.columns and data["species"].notna().all():
    print(f"\nNew-data accuracy: {accuracy_score(data['species'].astype(str), data['predicted_species']):.2%}")
    print("\nClassification report:")
    print(classification_report(
        data["species"].astype(str), data["predicted_species"],
        labels=list(encoder.classes_), target_names=list(encoder.classes_),
        zero_division=0
    ))
else:
    print("\nNo complete actual labels found; accuracy cannot be calculated.")

data.to_csv("prediction_results.csv", index=False)
print("\nSaved results to prediction_results.csv")

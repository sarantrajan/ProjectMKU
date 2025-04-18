import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Encode 'season'
label_encoder = LabelEncoder()
df["season"] = label_encoder.fit_transform(df["season"].str.lower())

# Features and target
X = df[["temperature", "humidity", "ph", "water availability", "season"]]
y = df["label"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("✅ Model Trained")
print("🔍 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model
with open("crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save season label encoder
with open("season_encoder.pkl", "wb") as enc_file:
    pickle.dump(label_encoder, enc_file)

print("💾 Model saved as 'crop_model.pkl'")
print("💾 LabelEncoder saved as 'season_encoder.pkl'")

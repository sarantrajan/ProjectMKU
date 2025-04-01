
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Encode categorical variable (Season)
label_encoder = LabelEncoder()
df["season"] = label_encoder.fit_transform(df["season"].str.lower())

# Define features and target
X = df.drop(columns=["label"])
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define individual models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
svm_model = SVC(probability=True, random_state=42)  # Enable probability for soft voting

# Create Voting Classifier (Soft Voting)
voting_clf = VotingClassifier(
    estimators=[("rf", rf_model), ("xgb", xgb_model), ("svm", svm_model)],
    voting="soft"  # Use soft voting for better performance
)

# Train Voting Classifier
voting_clf.fit(X_train, y_train)

# Save model
with open("crop_voting_model.pkl", "wb") as model_file:
    pickle.dump(voting_clf, model_file)

# Load trained model
with open("crop_voting_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        # Extract input values
        temperature = float(data.get("temperature", 0))
        humidity = float(data.get("humidity", 0))
        water_availability = float(data.get("water_availability", 0))
        ph = float(data.get("ph", 0))
        season = data.get("season", "").strip().lower()

        # Check if the season is valid
        if season not in label_encoder.classes_:
            return jsonify({"error": f"Invalid season: '{season}'. Expected values: {list(label_encoder.classes_)}"}), 400

        # Encode season
        season_encoded = label_encoder.transform([season])[0]

        # Prepare features for prediction
        features = np.array([[temperature, humidity, water_availability, ph, season_encoded]])

        # Predict crop
        prediction = model.predict(features)[0]

        return jsonify({"recommended_crop": prediction})

    except ValueError:
        return jsonify({"error": "Invalid input! Please enter valid numbers for all fields."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ----------------------------
# TRAINING SCRIPT
# ----------------------------
# Steps:
# 1. Load merged & balanced dataset
# 2. Split into train/test
# 3. Standardize features
# 4. Train RandomForest classifier
# 5. Evaluate performance
# 6. Save model to disk
# ----------------------------

DATA_FILE = "merged_gestures_balanced.csv"
MODEL_FILE = "gesture_model.pkl"

# Load dataset
df = pd.read_csv(DATA_FILE)

# Features & labels
X = df.drop("label", axis=1).values
y = df["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: normalize + classifier
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

print("[INFO] Training model...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("\n[RESULT] Classification Report:\n", classification_report(y_test, y_pred))
print("\n[CONFUSION MATRIX]\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(pipeline, MODEL_FILE)
print(f"[SAVED] Model stored as {MODEL_FILE}")

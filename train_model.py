"""
train_model.py
Run this locally (or let app.py auto-run it) to build career_model.joblib.
Usage: python train_model.py
"""
import os, sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "career_dataset.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH= os.path.join(MODEL_DIR, "career_model.joblib")
LE_PATH   = os.path.join(MODEL_DIR, "label_encoder.joblib")

FEATURES = ["math","logic","creativity","communication","leadership",
            "problem_solving","programming","data_analysis","design",
            "networking","management","writing"]

df = pd.read_excel(DATA_PATH)
print(f"Dataset: {df.shape}")

X  = df[FEATURES]
le = LabelEncoder()
y  = le.fit_transform(df["Career"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
    ))
])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cv     = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")

print(f"\nTest Accuracy : {acc:.4f}")
print(f"CV  Accuracy  : {cv.mean():.4f} ± {cv.std():.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(pipe, MODEL_PATH, compress=3)
joblib.dump(le,   LE_PATH,   compress=3)
sz = os.path.getsize(MODEL_PATH) / 1024
print(f"\n✅  Saved → {MODEL_PATH}  ({sz:.0f} KB)")
print(f"✅  Saved → {LE_PATH}")

"""
train_model.py  —  Student Performance Predictor
-------------------------------------------------
All files live in the same folder as this script.
Run with:
    python3 train_model.py
"""

import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── All paths are relative to THIS file ─────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(HERE, "students.csv")
MODEL_PKL = os.path.join(HERE, "performance_model.pkl")
LE_PKL    = os.path.join(HERE, "label_encoder.pkl")
MET_PKL   = os.path.join(HERE, "model_metrics.pkl")

FEATURES = ["study_hours", "attendance", "sleep_hours", "previous_score"]
TARGET   = "performance"

print("=" * 54)
print("   Student Performance Predictor -- Model Training")
print("=" * 54)

# ── Step 1: Generate dataset if students.csv is missing ─────────────────────
if not os.path.exists(CSV_PATH):
    print("\n  students.csv not found -- generating 250 rows ...")
    np.random.seed(42)

    def make_group(n, study_mu, att_mu, sleep_mu, score_mu, label):
        study = np.clip(np.random.normal(study_mu, 1.2, n),  0,  12)
        att   = np.clip(np.random.normal(att_mu,   8.0, n), 30, 100)
        sleep = np.clip(np.random.normal(sleep_mu, 1.0, n),  3,  12)
        score = np.clip(np.random.normal(score_mu, 8.0, n),  0, 100)
        return study, att, sleep, score, [label] * n

    s1,a1,sl1,sc1,p1 = make_group(80, 2.0, 58, 5.5, 42, "Low")
    s2,a2,sl2,sc2,p2 = make_group(90, 4.5, 73, 7.0, 62, "Medium")
    s3,a3,sl3,sc3,p3 = make_group(80, 7.5, 89, 8.0, 84, "High")

    df = pd.DataFrame({
        "study_hours"    : np.concatenate([s1,s2,s3]).round(1),
        "attendance"     : np.concatenate([a1,a2,a3]).round(1),
        "sleep_hours"    : np.concatenate([sl1,sl2,sl3]).round(1),
        "previous_score" : np.concatenate([sc1,sc2,sc3]).round(1),
        "performance"    : p1 + p2 + p3,
    }).sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(CSV_PATH, index=False)
    print(f"  Saved {len(df)} rows to students.csv")
else:
    print(f"\n  Found students.csv")

# ── Step 2: Load ─────────────────────────────────────────────────────────────
df    = pd.read_csv(CSV_PATH)
X     = df[FEATURES]
y     = df[TARGET]
print(f"  {len(df)} rows loaded | classes: {df[TARGET].value_counts().to_dict()}")

# ── Step 3: Encode labels ─────────────────────────────────────────────────────
le    = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\n  Label map: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── Step 4: Split ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── Step 5: Train ─────────────────────────────────────────────────────────────
print("\n  Training Random Forest ...")
model = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    min_samples_split=5, random_state=42, class_weight="balanced",
)
model.fit(X_train, y_train)

# ── Step 6: Evaluate ──────────────────────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)
report   = classification_report(y_test, y_pred, target_names=le.classes_,
                                 output_dict=True, zero_division=0)
print(f"\n  Accuracy: {accuracy * 100:.1f}%\n")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

importances = dict(zip(FEATURES, model.feature_importances_.tolist()))

# ── Step 7: Save — all .pkl files go into the SAME folder as this script ─────
metrics = {
    "accuracy"              : float(accuracy),
    "confusion_matrix"      : cm.tolist(),
    "classification_report" : report,
    "feature_importances"   : importances,
    "label_classes"         : list(le.classes_),
    "features"              : FEATURES,
}
joblib.dump(model,   MODEL_PKL)
joblib.dump(le,      LE_PKL)
joblib.dump(metrics, MET_PKL)

print(f"  Saved: performance_model.pkl")
print(f"  Saved: label_encoder.pkl")
print(f"  Saved: model_metrics.pkl")
print("\n  All files are in:", HERE)
print("\nDone! Now run:  streamlit run app.py\n")
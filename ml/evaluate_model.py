
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = Path(
    r"C:\Users\muska\Downloads\extention+of+z+alizadeh+sani+dataset\extention of Z-Alizadeh sani dataset.csv"
)

df = pd.read_csv(DATA_PATH)

model = joblib.load(ROOT / "model" / "cad_model.joblib")
encoder = joblib.load(ROOT / "model" / "cad_target_encoder.joblib")
print("Encoder classes:", encoder.classes_)
print("Model classes:", model.classes_)

X = df.drop(columns=["Cath", "LAD", "LCX", "RCA"])
y = encoder.transform(df["Cath"].astype(str).str.strip())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

pred = model.predict(X_test)
cad_index = list(model.classes_).index(
    int(list(encoder.classes_).index("CAD"))
)
proba = model.predict_proba(X_test)[:, cad_index]
cad_index = list(model.classes_).index(0)
cad_proba = model.predict_proba(X_test)[:, cad_index]
y_cad = (y_test == 0).astype(int)

print(
    "CAD ROC-AUC:",
    round(roc_auc_score(y_cad, cad_proba), 3),
)

print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("CAD Precision:", round(precision_score(y_test, pred, pos_label=0, zero_division=0), 3))
print("CAD Recall:", round(recall_score(y_test, pred, pos_label=0, zero_division=0), 3))
print("CAD F1-score:", round(f1_score(y_test, pred, pos_label=0, zero_division=0), 3))


print("\nClassification report:")
print(
    classification_report(
        y_test,
        pred,
        target_names=encoder.classes_,
        zero_division=0,
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred,
    display_labels=encoder.classes_,
)

plt.title("CAD Model - Holdout Confusion Matrix")
plt.tight_layout()
plt.savefig(ROOT / "model" / "confusion_matrix.png", dpi=150)
plt.close()

print("\nConfusion matrix saved successfully.")


from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(
    r"C:\Users\muska\Downloads\extention+of+z+alizadeh+sani+dataset\extention of Z-Alizadeh sani dataset.csv"
)
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

vessels = ["LAD", "LCX", "RCA"]
excluded = ["Cath", "LAD", "LCX", "RCA"]

for target in vessels:
    print(f"\n{'=' * 40}\nTarget: {target}\n{'=' * 40}")

    # Keep only records with a known target
    data = df.dropna(subset=[target]).copy()
    y_raw = data[target].astype(str).str.strip()

    print("Class counts:")
    print(y_raw.value_counts())

    if y_raw.nunique() != 2 or y_raw.value_counts().min() < 2:
        print(f"Skipping {target}: target is not suitable for binary stratified training.")
        continue

    X = data.drop(columns=excluded)

    # Convert target labels to 0/1
    classes = sorted(y_raw.unique())
    y = y_raw.map({classes[0]: 0, classes[1]: 1})

    categorical = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric = X.select_dtypes(include="number").columns.tolist()

    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\nClassification report:")
    print(classification_report(
        y_test, pred,
        labels=[0, 1],
        target_names=classes,
        zero_division=0
    ))

    if y_test.nunique() == 2:
        proba = model.predict_proba(X_test)[:, 1]
        print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))

    joblib.dump(
        {"model": model, "classes": classes, "features": X.columns.tolist()},
        MODEL_DIR / f"{target.lower()}_model.joblib"
    )
    print(f"Saved: model/{target.lower()}_model.joblib")

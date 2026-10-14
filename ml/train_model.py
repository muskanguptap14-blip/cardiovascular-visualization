
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance


# -------------------- PATHS --------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(
    r"C:\Users\muska\Downloads\extention+of+z+alizadeh+sani+dataset\extention of Z-Alizadeh sani dataset.csv"
)

MODEL_DIR = ROOT / "model"
ASSETS_DIR = ROOT / "assets"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------- LOAD DATA --------------------

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)

TARGET = "Cath"
EXCLUDED_COLUMNS = ["Cath", "LAD", "LCX", "RCA"]

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found. Available columns: {df.columns.tolist()}"
    )

# Overall CAD target
y_raw = df[TARGET].astype(str).str.strip()

# Keep only recognized binary labels.
# The dataset is expected to use CAD and Normal.
valid_labels = {"CAD", "Normal"}
if not set(y_raw.unique()).issubset(valid_labels):
    raise ValueError(
        f"Unexpected target labels: {sorted(y_raw.unique())}. "
        "Check the dataset's Cath column before training."
    )

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Do not use the target or vessel labels as input features.
X = df.drop(columns=EXCLUDED_COLUMNS, errors="ignore").copy()

# Treat categorical values consistently.
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col] = X[col].apply(
        lambda value: value.strip() if isinstance(value, str) else value
    )

# Convert boolean columns to categorical strings.
for col in X.select_dtypes(include=["bool"]).columns:
    X[col] = X[col].astype(str)

numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = [
    col for col in X.columns if col not in numeric_columns
]

print("Numeric features:", len(numeric_columns))
print("Categorical features:", len(categorical_columns))
print("Target classes:", list(label_encoder.classes_))


# -------------------- PREPROCESSING --------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])


# -------------------- TRAIN / TEST SPLIT --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# -------------------- TRAIN MODEL --------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    ),
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Explicitly treat CAD as the positive class.
cad_class_index = list(label_encoder.transform(label_encoder.classes_)).index(
    label_encoder.transform(["CAD"])[0]
)
y_test_cad = (y_test == cad_class_index).astype(int)
predictions_cad = (predictions == cad_class_index).astype(int)
cad_probabilities = probabilities[:, cad_class_index]

print("\n------ CAD TEST METRICS ------")
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(
    "CAD Precision:",
    round(precision_score(y_test_cad, predictions_cad, zero_division=0), 3),
)
print(
    "CAD Recall:",
    round(recall_score(y_test_cad, predictions_cad, zero_division=0), 3),
)
print(
    "CAD F1:",
    round(f1_score(y_test_cad, predictions_cad, zero_division=0), 3),
)

if len(set(y_test_cad)) == 2:
    print(
        "CAD ROC-AUC:",
        round(roc_auc_score(y_test_cad, cad_probabilities), 3),
    )
else:
    print("CAD ROC-AUC: unavailable (test set has only one class)")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)


# -------------------- CONFUSION MATRIX --------------------

cm = confusion_matrix(
    y_test,
    predictions,
    labels=range(len(label_encoder.classes_)),
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_,
)

display.plot(cmap="Blues", values_format="d")
plt.title("CAD Classification — Test Set")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "confusion_matrix.png", dpi=160)
plt.close()

print("Saved:", ASSETS_DIR / "confusion_matrix.png")


# -------------------- PERMUTATION FEATURE IMPORTANCE --------------------
# This evaluates the full pipeline on the held-out test set.

print("\nCalculating permutation feature importance...")

importance_result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance_result.importances_mean,
    "Std": importance_result.importances_std,
}).sort_values("Importance", ascending=False)

importance_df.to_csv(
    ASSETS_DIR / "feature_importance.csv",
    index=False,
)

top_importance = (
    importance_df.head(15)
    .sort_values("Importance", ascending=True)
)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(
    top_importance["Feature"],
    top_importance["Importance"],
    xerr=top_importance["Std"],
)
ax.set_title("Top 15 Features — Permutation Importance")
ax.set_xlabel("Decrease in test accuracy when feature is shuffled")
ax.set_ylabel("Input feature")
fig.tight_layout()
fig.savefig(ASSETS_DIR / "feature_importance.png", dpi=160)
plt.close(fig)

print("\nTop feature importance:")
print(importance_df.head(15).to_string(index=False))
print("Saved:", ASSETS_DIR / "feature_importance.png")
print("Saved:", ASSETS_DIR / "feature_importance.csv")


# -------------------- SAVE MODEL AND ENCODER --------------------

joblib.dump(model, MODEL_DIR / "cad_model.joblib")
joblib.dump(label_encoder, MODEL_DIR / "cad_target_encoder.joblib")

# Save the feature schema used during training.
joblib.dump(X.columns.tolist(), MODEL_DIR / "cad_feature_columns.joblib")

print("\nSaved model:", MODEL_DIR / "cad_model.joblib")
print("Saved encoder:", MODEL_DIR / "cad_target_encoder.joblib")
print("Saved feature schema:", MODEL_DIR / "cad_feature_columns.joblib")
print("\nTraining and evaluation completed.")

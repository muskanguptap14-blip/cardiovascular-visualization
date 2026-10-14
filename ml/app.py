
from pathlib import Path
from datetime import datetime
import base64
import html
import os
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# -------------------- CONFIG --------------------

st.set_page_config(
    page_title="Cardiovascular Risk Dashboard",
    page_icon="❤️",
    layout="wide",
)

st.title("❤️ Cardiovascular Risk Visualization & Prediction")
st.caption(
    "Machine-learning hackathon prototype | "
    "Educational use only — not a medical diagnosis."
)


# -------------------- PATHS --------------------

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
ASSETS_DIR = ROOT / "assets"

MODEL_PATH = MODEL_DIR / "cad_model.joblib"
ENCODER_PATH = MODEL_DIR / "cad_target_encoder.joblib"
MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"
HEART_PATH = ASSETS_DIR / "heart.glb"

DATA_PATH = Path(
    r"C:\Users\muska\Downloads\extention+of+z+alizadeh+sani+dataset\extention of Z-Alizadeh sani dataset.csv"
)


# -------------------- LOAD DATA AND MODELS --------------------

if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
    st.error("CAD model files not found in the model folder.")
    st.stop()

if not DATA_PATH.exists():
    st.error(
        f"Dataset not found:\n{DATA_PATH}\n\n"
        "Update DATA_PATH near the top of ml/app.py."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
df = pd.read_csv(DATA_PATH)

excluded = ["Cath", "LAD", "LCX", "RCA"]
X = df.drop(columns=excluded, errors="ignore")


# -------------------- MODEL PERFORMANCE --------------------
# These are the previously reported project metrics.
# Verify them against your held-out evaluation before presenting.

st.header("Model Performance")

c1, c2, c3 = st.columns(3)
c1.metric("Reported weighted F1", "85%")
c2.metric("Reported ROC-AUC", "88.4%")
c3.metric("Dataset records", f"{len(df):,}")

if MATRIX_PATH.exists():
    st.image(
        str(MATRIX_PATH),
        caption="Confusion matrix from model evaluation",
        use_container_width=True,
    )

st.caption(
    "Confirm the displayed metrics against your evaluation script "
    "before treating them as final benchmark results."
)

st.divider()


# -------------------- PATIENT INPUTS --------------------

st.header("Patient Risk Prediction")
st.write(
    "Enter feature values in the sidebar, then click Predict."
)

st.sidebar.header("Patient Features")
st.sidebar.caption("Values are limited to the dataset's observed ranges.")

input_data = {}

for col in X.columns:
    series = X[col]

    if pd.api.types.is_numeric_dtype(series):
        median_value = float(series.median())
        minimum = float(series.min())
        maximum = float(series.max())

        if pd.isna(median_value):
            st.error(f"Feature '{col}' contains no usable numeric values.")
            st.stop()

        if minimum == maximum:
            input_data[col] = st.sidebar.number_input(
                col,
                value=median_value,
                disabled=True,
            )
        else:
            input_data[col] = st.sidebar.number_input(
                col,
                min_value=minimum,
                max_value=maximum,
                value=min(max(median_value, minimum), maximum),
            )
    else:
        options = series.dropna().unique().tolist()

        if not options:
            st.error(f"No available values for feature: {col}")
            st.stop()

        input_data[col] = st.sidebar.selectbox(col, options)


# -------------------- PREDICTION --------------------

if st.button("Predict", type="primary", use_container_width=True):

    patient = pd.DataFrame([input_data], columns=X.columns)

    prediction = model.predict(patient)[0]
    probabilities = model.predict_proba(patient)[0]
    predicted_label = encoder.inverse_transform([prediction])[0]

    st.header("Prediction Results")

    st.subheader("Overall CAD Prediction")
    st.metric("Predicted class", str(predicted_label))

    probability_df = pd.DataFrame({
        "Class": encoder.classes_,
        "Probability": probabilities,
    })

    probability_df["Probability (%)"] = (
        probability_df["Probability"] * 100
    ).round(2)

    st.subheader("Estimated Class Probabilities")
    st.bar_chart(
        probability_df.set_index("Class")["Probability"]
    )
    st.dataframe(
        probability_df[["Class", "Probability (%)"]],
        hide_index=True,
        use_container_width=True,
    )

    st.warning(
        "Experimental model output only. This is not a validated "
        "clinical risk score and must not be used to diagnose or treat "
        "a person."
    )

    # -------------------- VESSEL PREDICTIONS --------------------

    st.divider()
    st.header("Coronary Vessel Predictions")

    vessel_columns = st.columns(3)
    vessel_results = []

    for column, vessel in zip(
        vessel_columns, ["LAD", "LCX", "RCA"]
    ):
        artifact_path = MODEL_DIR / f"{vessel.lower()}_model.joblib"

        if not artifact_path.exists():
            st.error(f"Missing model file: {artifact_path.name}")
            st.stop()

        artifact = joblib.load(artifact_path)

        vessel_model = artifact["model"]
        features = list(artifact["features"])
        classes = list(artifact["classes"])

        missing_features = [
            feature for feature in features
            if feature not in patient.columns
        ]

        if missing_features:
            st.error(
                f"{vessel} model needs missing features: "
                f"{missing_features}"
            )
            st.stop()

        vessel_patient = patient[features]
        raw_prediction = vessel_model.predict(vessel_patient)[0]
        vessel_probabilities = vessel_model.predict_proba(
            vessel_patient
        )[0]

        # Handle models whose predictions are encoded as class indices.
        predicted_vessel_class = classes[int(raw_prediction)]

        stenosis_probability = (
            float(vessel_probabilities[classes.index("Stenotic")])
            if "Stenotic" in classes
            else None
        )

        with column:
            st.subheader(vessel)
            st.metric("Predicted status", predicted_vessel_class)

            if stenosis_probability is not None:
                st.metric(
                    "Estimated stenosis probability",
                    f"{stenosis_probability:.1%}",
                )

        vessel_results.append({
            "Vessel": vessel,
            "Predicted status": predicted_vessel_class,
            "Stenosis probability": stenosis_probability,
        })

    vessel_df = pd.DataFrame(vessel_results)

    st.subheader("Vessel Comparison")
    st.dataframe(vessel_df, hide_index=True, use_container_width=True)

    if vessel_df["Stenosis probability"].notna().any():
        chart_df = vessel_df.dropna(
            subset=["Stenosis probability"]
        ).set_index("Vessel")

        st.bar_chart(chart_df["Stenosis probability"])

    # -------------------- INTERACTIVE 3D HEART --------------------

    st.divider()
    st.header("Interactive 3D Heart Viewer")
    st.caption(
        "Rotate and zoom the heart model. Vessel predictions are shown "
        "separately; this model does not anatomically map predictions "
        "onto LAD, LCX, or RCA."
    )

    if HEART_PATH.exists():
        try:
            heart_base64 = base64.b64encode(
                HEART_PATH.read_bytes()
            ).decode("utf-8")

            viewer_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
              <meta name="viewport"
                    content="width=device-width, initial-scale=1">
              <script type="module"
                src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js">
              </script>
              <style>
                body {{
                  margin: 0;
                  background: #171923;
                  font-family: sans-serif;
                }}
                model-viewer {{
                  width: 100%;
                  height: 520px;
                  background: #171923;
                  --poster-color: #171923;
                }}
              </style>
            </head>
            <body>
              <model-viewer
                src="data:model/gltf-binary;base64,{heart_base64}"
                camera-controls
                auto-rotate
                shadow-intensity="1"
                exposure="1"
                environment-image="neutral"
                alt="Interactive 3D anatomical heart model">
              </model-viewer>
            </body>
            </html>
            """

            components.html(viewer_html, height=535, scrolling=False)

        except Exception as exc:
            st.error(f"Could not load the 3D heart model: {exc}")
    else:
        st.warning(
            "3D heart file not found. Expected location: assets/heart.glb"
        )



   
    # -------------------- DOWNLOADABLE REPORT --------------------

    st.divider()
    st.header("Patient Prediction Report")

    report_rows = [{
        "Measure": "Overall CAD",
        "Prediction": str(predicted_label),
        "Probability": round(
            float(probabilities[list(encoder.classes_).index("CAD")]),
            4,
        ),
    }]

    for result in vessel_results:
        report_rows.append({
            "Measure": result["Vessel"],
            "Prediction": result["Predicted status"],
            "Probability": (
                round(float(result["Stenosis probability"]), 4)
                if result["Stenosis probability"] is not None
                else None
            ),
        })

    report_df = pd.DataFrame(report_rows)

    st.dataframe(report_df, use_container_width=True)

# ---------------- FEATURE IMPORTANCE ----------------


# ---------------- FEATURE IMPORTANCE ----------------

st.divider()
st.header("What Influences the Model?")

importance_path = "assets/feature_importance.png"

if os.path.exists(importance_path):
    st.image(
        importance_path,
        caption="Permutation feature importance on the test set",
        use_container_width=True,
    )
else:
    st.warning("Feature importance chart not found.")

st.caption(
    "Higher values indicate a larger effect on test-set accuracy "
    "when a feature is shuffled. This does not establish causation."
)


# -------------------- DISCLAIMER --------------------

st.divider()
st.caption(
    "Educational prototype only. Model performance does not establish "
    "clinical validity. Consult qualified healthcare professionals "
    "for medical decisions."
)



# Cardiovascular Risk Visualization & Prediction

## Overview

A machine-learning prototype for exploring coronary artery disease (CAD) predictions and vessel-level predictions through an interactive web dashboard with a 3D heart visualization.

## Features

- Interactive Streamlit dashboard for clinical feature inputs.
- CAD prediction with model-generated class probabilities.
- LAD, LCX, and RCA vessel-level predictions.
- Interactive 3D heart model.
- Model evaluation metrics and permutation feature importance.
- Downloadable patient prediction report.

## Dataset and Preprocessing

The project uses the Z-Alizadeh Sani dataset, containing 303 records and 59 columns in the current dataset file.

Preprocessing includes separating the target from input features, handling numeric and categorical variables, imputing missing values, scaling numeric features, encoding categorical features, and splitting the data into training and test sets.

The overall CAD model excludes `Cath`, `LAD`, `LCX`, and `RCA` from its input features to prevent direct target leakage.

## Model Architecture

The CAD prediction pipeline uses preprocessing followed by a Random Forest classifier with 300 trees and class balancing.

The dataset is split into 80% training and 20% testing, using stratification and random seed 42.

Permutation importance is used to estimate how shuffling individual input features affects test-set accuracy.

## Preliminary Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 85.2% |
| CAD precision | 87.0% |
| CAD recall | 93.0% |
| CAD F1-score | 89.9% |
| CAD ROC-AUC | 88.4% |

These results come from a single 61-record test split. They are preliminary and do not establish clinical validity or generalization to new populations.

## Running the Application

1. Install Python and create a virtual environment.
2. Install the dependencies listed in `requirements.txt`.
3. Confirm that the dataset and required model files are in their expected locations.
4. Run the application from the project root:

   `python -m streamlit run ml/app.py`

5. Open the local URL displayed in the terminal.

## Limitations and Safety

This project is a research and educational prototype, not a medical diagnostic tool. Predictions require independent validation before any clinical use. Model-generated probabilities are not measured percentages of arterial narrowing. Vessel-level predictions do not identify patient-specific anatomical stenosis locations.

## Project Structure

- `ml/` — training, evaluation, and dashboard code
- `model/` — saved model pipelines and encoders
- `assets/` — 3D assets and evaluation visualizations
- `data/` — dataset files, subject to applicable data-sharing restrictions

## Future Improvements

- Independent validation and more robust evaluation.
- Better calibration and uncertainty reporting.
- Clinically reviewed explanations.
- Improved anatomical mapping and vessel-specific visualization.

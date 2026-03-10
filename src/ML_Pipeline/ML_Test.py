"""
=============================================================================
RAPID HYPERPARAMETER TUNER & CLINICAL INFERENCE TESTING
=============================================================================
Description:
This script evaluates the fully trained Random Forest model on the unseen 
continuous holdout dataset (Patients 21-24). It allows for rapid tuning 
of the confidence threshold and temporal median filter size to optimise 
the balance between early seizure detection (Recall) and false positive 
reduction (Precision).

By iterating through the data file-by-file, it ensures the median filter 
does not bleed across separate EEG recordings, simulating real-world 
continuous hospital monitoring.
=============================================================================
"""

import pandas as pd
import numpy as np
import joblib
from scipy.signal import medfilt
from sklearn.metrics import classification_report


# 1. HYPERPARAMETERS TO TUNE

# Lower = More aggressive at catching seizures (higher recall, lower precision)
CONFIDENCE_THRESHOLD = 0.40  
# Must be an odd number (3, 5, 7, 9). Represents the sliding temporal window.
MEDIAN_FILTER_SIZE = 5       


# 2. LOAD DATA AND MODEL

print("Loading continuous test data and model...")
df_test = pd.read_csv("generalised_testing_continuous.csv")
rf_classifier = joblib.load('chb_master_model.pkl')


# 3. APPLY THE OPTIMISED BPSO MASK - Obtained from ParticleSwarm.py

winning_mask = np.array([
    1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
    1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1,
    1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0,
    0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0
])

# Arrays to hold the massive combined results across all files
all_y_true = []
all_y_pred_filtered = []


# 4. PROCESS FILE BY FILE

unique_files = df_test['Filename'].unique()

for file in unique_files:
    # Isolate just this specific file so the median filter doesn't bleed across different recordings
    file_data = df_test[df_test['Filename'] == file]
    
    # Extract features (X) and ground truth labels (y)
    X_raw = file_data.iloc[:, 1:-1].values # Skip the Filename and Label columns
    y_true = file_data.iloc[:, -1].values
    
    # Apply the optimised mask to drop useless features
    X_inference = X_raw[:, winning_mask == 1]
    
    # Predict Probabilities instead of hard binary choices
    probabilities = rf_classifier.predict_proba(X_inference)[:, 1]
    
    # Apply the custom confidence threshold
    raw_predictions = (probabilities >= CONFIDENCE_THRESHOLD).astype(int)
    
    # Apply Temporal Median Filter to erase short, isolated false positive artifacts
    filtered_predictions = medfilt(raw_predictions, kernel_size=MEDIAN_FILTER_SIZE)
    
    # Append to master evaluation lists
    all_y_true.extend(y_true)
    all_y_pred_filtered.extend(filtered_predictions)


# 5. GENERATE CLINICAL REPORT

print(f"\n====================================================")
print(f" TUNING REPORT: Conf >= {CONFIDENCE_THRESHOLD} | Filter = {MEDIAN_FILTER_SIZE}")
print(f" Total Continuous Epochs Evaluated: {len(all_y_true)}")
print(f"====================================================")

print(classification_report(all_y_true, all_y_pred_filtered, target_names=["Normal (0)", "Seizure (1)"]))
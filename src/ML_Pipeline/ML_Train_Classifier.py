"""
=============================================================================
MASTER RANDOM FOREST CLASSIFIER TRAINING
=============================================================================
Description:
This script acts as the final stage of the model training pipeline. It takes 
the air-gapped training dataset (Patients 1-20) and applies the optimal feature
mask discovered by the Particle Swarm (BPSO). 

By dropping the 96 "noisy" features and keeping the 96 "winning" features, 
it trains a robust 100-tree Random Forest Classifier. It uses a balanced 
class weight to ensure the algorithm pays strict attention to the rare seizure 
events. Finally, it exports the fully trained model as a .pkl file, 
which can be deployed to the testing scripts.
=============================================================================
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier


# 1. LOAD THE TRAINING DATA

print("Loading Training Data (Patients 1-20)...")
df_train = pd.read_csv("generalised_training_subset.csv")

# Split into Features (X) and Labels (y)
X_train_full = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values


# 2. APPLY THE OPTIMIZED BPSO MASK - Obtained from ParticleSwarm.py
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


# Drop all the useless columns identified by the Swarm
X_train_optimised = X_train_full[:, winning_mask == 1]
print(f"Features optimized: 192 -> {X_train_optimised.shape[1]}")


# 3. BUILD AND TRAIN THE FINAL CLASSIFIER

print("\nPlanting the Random Forest...")


rf_classifier = RandomForestClassifier(
    n_estimators=100,        # 100 independent decision trees for maximum stability
    max_depth=None,          # Allow trees to learn deep, complex entropy patterns
    class_weight='balanced', # Crucial mathematical adjustment for 1:3 imbalanced medical data
    n_jobs=-1,               # Use all available CPU cores to speed up training
    random_state=42          # Lock the seed for exact reproducibility
)

print("Training the model on Patients 1-20...")
rf_classifier.fit(X_train_optimised, y_train)


# 4. EXPORT THE TRAINED MODEL
model_filename = 'chb_master_model.pkl'
joblib.dump(rf_classifier, model_filename)

print(f"\n SUCCESS: Model trained and saved as '{model_filename}'")
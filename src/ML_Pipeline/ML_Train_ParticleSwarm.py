"""
=============================================================================
BINARY PARTICLE SWARM OPTIMIZATION (BPSO) FOR EEG FEATURE SELECTION
=============================================================================
Description:
This script performs evolutionary feature selection on the extracted EEG 
training dataset. It uses a Binary Particle Swarm Optimization (BPSO) algorithm 
enhanced with Time-Varying Acceleration Coefficients (TVAC), velocity clamping, 
and genetic mutation to escape local minimums and find the global optimal 
subset of VMD-Entropy features. 

It evaluates thousands of feature masks by training a lightweight Random Forest 
(10 trees) and scoring it based on the F1-Score to perfectly handle the 
1:3 class imbalance. The final output is the optimal binary array mask to be 
applied to the final Random Forest Classifier.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# 1. LOAD AND PREPARE DATA

# Load the air-gapped training dataset (Patients 1-20 only)
df = pd.read_csv("generalised_training_subset.csv")

# Isolate features (X) and labels (Y)
X_data = df.iloc[:, :-1].values 
Y_data = df.iloc[:, -1].values 

# 30% holdout purely for the Swarm's internal objective function
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=42)
n_features = X_train.shape[1]


# 2. PSO HYPERPARAMETERS

# Inertia bounds (starts high for exploration, ends low for exploitation)
w_max, w_min = 0.9, 0.4

# Time-Varying Acceleration Coefficients (TVAC)
c1_max, c1_min = 2.5, 0.5
c2_min, c2_max = 0.5, 2.5

V_max = 2.0           # Strict velocity clamp to prevent mathematical certainty
mutation_rate = 0.1   # 10% chance to randomly flip bits and break out of ruts
n_particles = 50      # Swarm size
n_iterations = 50     # Total generations

# Initialize the Swarm's positions and velocities
X = np.random.randint(2, size=(n_particles, n_features))
V = np.random.uniform(-1, 1, (n_particles, n_features))


# 3. OBJECTIVE FUNCTION
def f(particle_mask):
    """
    Evaluates a specific binary mask by applying it to the data, 
    training a mini Random Forest, and returning the F1-Score.
    """
    # Penalty: If the swarm turns off every single feature, score is 0
    if np.sum(particle_mask) == 0: 
        return 0.0
    
    # Apply the binary mask to the dataset
    X_train_masked = X_train[:, particle_mask == 1]
    X_test_masked = X_test[:, particle_mask == 1]
    
    # Train the mini model
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train_masked, Y_train)
    
    # Score based on F1 to penalise models that ignore seizures
    predictions = clf.predict(X_test_masked)
    return f1_score(Y_test, predictions, average='macro')

def sigmoid(x):
    """ Converts continuous velocity into a probability strictly between 0 and 1 """
    x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))

# Evaluate the initial random starting positions
pbest = X.copy()
pbest_obj = np.array([f(particle) for particle in X]) 
gbest = pbest[pbest_obj.argmax()].copy()
gbest_obj = pbest_obj.max()

# 4. THE UPDATE LOOP
def update_particles(iteration):
    global X, V, pbest, pbest_obj, gbest, gbest_obj
    
    # 1. Calculate decaying parameters
    w = w_max - ((w_max - w_min) * (iteration / n_iterations))
    c1 = c1_max - ((c1_max - c1_min) * (iteration / n_iterations))
    c2 = c2_min + ((c2_max - c2_min) * (iteration / n_iterations))
    
    r1 = np.random.rand(n_particles, n_features)
    r2 = np.random.rand(n_particles, n_features)
    
    # 2. Update Velocity
    V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
    
    # 3. Clamp velocity to enforce continuous exploration
    V = np.clip(V, -V_max, V_max)
    probabilities = sigmoid(V)
    
    # 4. Determine new binary positions based on velocity probability
    rand_matrix = np.random.rand(n_particles, n_features)
    X_new = (rand_matrix < probabilities).astype(int)
    
    # 5. Apply Genetic Mutation
    mutations = (np.random.rand(n_particles, n_features) < mutation_rate).astype(int)
    X = np.bitwise_xor(X_new, mutations)
    
    # 6. Evaluate the new positions and update personal/global bests
    for i in range(n_particles):
        obj = f(X[i])
        if obj > pbest_obj[i]:
            pbest[i] = X[i].copy()
            pbest_obj[i] = obj
            if obj > gbest_obj:
                gbest = X[i].copy()
                gbest_obj = obj

# 5. EXECUTION AND REPORTING
if __name__ == "__main__":
    print(f"Starting BPSO Feature Selection (Starting Features: {n_features})")
    
    # Run the evolutionary loop
    for it in range(n_iterations):
        update_particles(it)
        active_features = np.sum(gbest)
        print(f"Iter {it+1}/{n_iterations} | Macro F1-Score: {gbest_obj:.4f} | Features Kept: {active_features}")
        
    print("\n====================================================")
    print("                 PSO OPTIMIZATION COMPLETE            ")
    print("====================================================")
    print(f"Final Best F1-Score: {gbest_obj:.4f}")
    print(f"Total Features Reduced: {n_features} -> {np.sum(gbest)}")
    
    print("\nBest Feature Mask:")
    print(np.array2string(gbest, separator=' ', max_line_width=80))
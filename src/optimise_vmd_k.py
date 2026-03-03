import mne
import os
import re
import numpy as np
import random
from vmdpy import VMD

# ========================================== CONFIGURATION  ==========================================
DATASET_PATH = "chbmit_dataset" 
CHANNEL_NAME = 'FP1-F7'
EPOCH_DURATION = 2.0
FS = 256.0
NUM_SAMPLES = 10

# ========================================== 1. TEXT PARSER & RANDOM FILE SELECTOR ==========================================
def get_random_seizure_file(dataset_path):
    """
    Scans the CHB-MIT dataset, reads the summary text files, and returns 
    a random EDF file path along with its seizure start and end times.
    """
    patients = [d for d in os.listdir(dataset_path) if d.startswith('chb') and os.path.isdir(os.path.join(dataset_path, d))]
    random.shuffle(patients) # Randomise patient order
    
    for patient in patients:
        summary_path = os.path.join(dataset_path, patient, f"{patient}-summary.txt")
        if not os.path.exists(summary_path):
            continue
            
        with open(summary_path, 'r') as file:
            lines = file.readlines()
            
        current_file = None
        for i, line in enumerate(lines):
            # Track the current file being described
            if line.startswith("File Name:"):
                current_file = line.split(":")[1].strip()
                
            # If the file has a seizure, extract the times!
            elif line.startswith("Number of Seizures in File:") and int(line.split(":")[1].strip()) > 0:
                try:
                    start_line = lines[i+1]
                    end_line = lines[i+2]
                    
                    # Use regex to pull out the integer seconds from the text
                    start_sec = int(re.findall(r'\d+', start_line)[-1])
                    end_sec = int(re.findall(r'\d+', end_line)[-1])
                    
                    full_path = os.path.join(dataset_path, patient, current_file)
                    
                    # Ensure the EDF file actually exists on the hard drive
                    if os.path.exists(full_path):
                        print(f"--> Selected: {patient}/{current_file}")
                        print(f"--> Seizure window: {start_sec}s to {end_sec}s")
                        return full_path, start_sec, end_sec
                except Exception as e:
                    continue
                    
    raise FileNotFoundError("Could not find any valid seizure files in the dataset folder.")

# ========================================== 2. VMD OPTIMISATION ALGORITHM ==========================================
def find_optimal_k(signal, fs, max_k=8, freq_threshold_hz=2.0):
    """Finds the optimal K value by ensuring modes don't split too closely."""
    alpha, tau, DC, init, tol = 2000, 0.0, 0, 1, 1e-7
    optimal_k = 2  
    
    for k_test in range(2, max_k + 1):
        u, u_hat, omega = VMD(signal, alpha, tau, k_test, DC, init, tol)
        final_omegas_hz = omega[-1, :] * fs
        final_omegas_hz.sort()
        distances = np.diff(final_omegas_hz)
        
        if np.any(distances < freq_threshold_hz):
            optimal_k = k_test - 1
            break
        else:
            optimal_k = k_test
            
    return optimal_k

# ========================================== 3. MAIN PIPELINE SCRIPT ==========================================
def main():
    # 1. Automatically find a file with a seizure
    file_path, start_sec, end_sec = get_random_seizure_file(DATASET_PATH)
    
    # 2. Load and Filter the Data
    print("\nLoading and filtering EEG data...")
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose='ERROR')
    raw.notch_filter(freqs=[60.0], verbose='ERROR')
    raw.filter(l_freq=1.0, h_freq=40.0, verbose='ERROR')
    
    # 3. Epoch Generation
    print("Chopping continuous data into epochs...")
    epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_DURATION, preload=True, verbose='ERROR')
    data_matrix = epochs.get_data()
    total_epochs = data_matrix.shape[0]
    
    # 4. Calculate Epoch Indices
    # Convert seconds to epoch indices
    seizure_start_idx = int(start_sec // EPOCH_DURATION)
    seizure_end_idx = int(end_sec // EPOCH_DURATION)
    
    # Define the ranges, ensuring we don't go out of bounds
    seizure_epoch_indices = list(range(seizure_start_idx, min(seizure_end_idx + 1, total_epochs)))
    
    safe_normal_end = max(0, seizure_start_idx - 10)
    normal_epoch_indices = list(range(0, safe_normal_end))
    
    if len(normal_epoch_indices) < NUM_SAMPLES or len(seizure_epoch_indices) == 0:
        print("\n[!] The randomly selected file doesn't have enough safe epochs to test properly.")
        print("Please run the script again to pick a different file!")
        return

    # 5. Randomly Sample
    random.seed(42)
    test_normal = random.sample(normal_epoch_indices, min(NUM_SAMPLES, len(normal_epoch_indices)))
    test_seizure = random.sample(seizure_epoch_indices, min(NUM_SAMPLES, len(seizure_epoch_indices)))
    
    print(f"\nTesting {len(test_normal)} Normal epochs and {len(test_seizure)} Seizure epochs...")
    
    ch_idx = epochs.ch_names.index(CHANNEL_NAME)
    
    # 6. Analyse Normal Epochs
    print("\n--- ANALYZING NORMAL EPOCHS ---")
    normal_k_results = []
    for i, idx in enumerate(test_normal):
        signal = data_matrix[idx, ch_idx, :]
        k = find_optimal_k(signal, FS)
        normal_k_results.append(k)
        print(f"Normal Epoch {i+1}/{len(test_normal)} -> Optimal K: {k}")

    # 7. Analyse Seizure Epochs
    print("\n--- ANALYZING SEIZURE EPOCHS ---")
    seizure_k_results = []
    for i, idx in enumerate(test_seizure):
        signal = data_matrix[idx, ch_idx, :]
        k = find_optimal_k(signal, FS)
        seizure_k_results.append(k)
        print(f"Seizure Epoch {i+1}/{len(test_seizure)} -> Optimal K: {k}")

    # 8. Print Final Verdict
    print("\n=============================================")
    print("           FINAL OPTIMAL K VERDICT             ")
    print("=============================================")
    print(f"Normal Epochs (Average K):  {np.mean(normal_k_results):.2f} (Max: {max(normal_k_results)})")
    print(f"Seizure Epochs (Average K): {np.mean(seizure_k_results):.2f} (Max: {max(seizure_k_results)})")
    
    recommended_k = max(max(normal_k_results), max(seizure_k_results))
    print(f"\n>>> Recommended Hardcoded K for ML Pipeline: K = {recommended_k} <<<")


if __name__ == "__main__":
    main()
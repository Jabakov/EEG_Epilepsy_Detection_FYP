import mne
import os
import re
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ==========================================
# 1. HELPER: PARSE TRUE SEIZURE WINDOWS
# ==========================================
def get_seizure_windows(summary_file_path, target_edf_name):
    seizure_windows = []
    if not os.path.exists(summary_file_path): return seizure_windows
    with open(summary_file_path, 'r') as file: lines = file.readlines()
    
    found_target = False
    num_seizures = 0
    for i, line in enumerate(lines):
        if line.startswith("File Name:") and target_edf_name in line:
            found_target = True
            continue 
        if found_target:
            if line.startswith("File Name:"): break
            if line.startswith("Number of Seizures in File:"):
                num_seizures = int(re.findall(r'\d+', line)[-1])
                if num_seizures == 0: break 
            elif num_seizures > 0 and line.startswith("Seizure") and "Start Time" in line:
                start_sec = int(re.findall(r'\d+', line)[-1])
                end_sec = int(re.findall(r'\d+', lines[i+1])[-1])
                seizure_windows.append((start_sec, end_sec))
    return seizure_windows

if __name__ == "__main__":
    
    # Global Testing Parameters
    epoch_duration = 2.0
    sampling_rate = 128.0
    TEST_PATIENTS = [21, 22, 23, 24] # The Unseen Vault
    
    # Load the Trained Model
    print("Loading Trained Spatiotemporal Transformer...")
    model = tf.keras.models.load_model("best_spatiotemporal_model.keras")
    
    # Global Trackers
    global_y_true = []
    global_y_pred = []
    global_total_epochs = 0
    
    print("\n====================================================")
    print("      INITIATING PATIENT-INDEPENDENT EVALUATION       ")
    print("====================================================")

    for patient_id_value in TEST_PATIENTS: 
        patient_id = f"{patient_id_value:02d}" 
        patient_folder = f"chbmit_dataset/chb{patient_id}"
        summary_file = os.path.join(patient_folder, f"chb{patient_id}-summary.txt")

        if not os.path.exists(patient_folder):
            continue

        edf_files = sorted([f for f in os.listdir(patient_folder) if f.endswith('.edf')])
        
        # Patient-Level Trackers
        patient_X = []
        patient_Y = []

        print(f"\n[Extracting Clinical Timeline for Patient {patient_id}...]")

        for edf_name in edf_files:
            file_path = os.path.join(patient_folder, edf_name)
            seizures = get_seizure_windows(summary_file, edf_name)
            
            # Load & Filter (Identical to training, no data leaks!)
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose='ERROR')
            raw.notch_filter(freqs=[60.0], verbose='ERROR')
            raw.filter(l_freq=1.0, h_freq=40.0, verbose='ERROR')
            
            channel_mapping = {ch: ch.replace('EEG ', '').replace(' ', '').upper() for ch in raw.ch_names}
            raw.rename_channels(channel_mapping)
            
            target_bases = ['FP1-F7', 'T7-P7', 'FP2-F8', 'T8-P8', 'F3-C3', 'C3-P3', 'F4-C4', 'C4-P4']
            actual_channels = [ch for base in target_bases for ch in raw.ch_names if base in ch]
            
            if len(actual_channels) == 8:
                raw.pick_channels(actual_channels, verbose='ERROR')
                raw.reorder_channels(actual_channels)
            else:
                continue 
            
            raw.resample(sampling_rate, verbose='ERROR')
            data_matrix = raw.get_data() 
            
            # STRICT CHRONOLOGICAL WINDOWING (No overlap, purely sequential)
            window_size = int(epoch_duration * sampling_rate)
            seizure_samples = [(start * sampling_rate, end * sampling_rate) for start, end in seizures]
            
            current_start = 0
            while current_start + window_size <= data_matrix.shape[1]:
                current_end = current_start + window_size
                
                is_seizure = False
                for (s_start, s_end) in seizure_samples:
                    if current_end > s_start and current_start < s_end:
                        is_seizure = True
                        break
                
                patient_X.append(data_matrix[:, current_start:current_end])
                patient_Y.append(1 if is_seizure else 0)
                
                current_start += window_size # Move forward exactly 2 seconds

        # ==========================================
        # PATIENT-LEVEL EVALUATION
        # ==========================================
        if len(patient_X) == 0: continue
        
        X_test = np.array(patient_X)
        y_true = np.array(patient_Y)
        
        print(f"  -> Diagnosing {len(X_test)} chronological epochs...")
        # Get raw probabilities, then convert to strict 0 or 1 labels
        y_pred_probs = model.predict(X_test, batch_size=64, verbose=0)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        
        # Calculate Patient Metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        patient_hours = len(X_test) * epoch_duration / 3600.0
        patient_far = fp / patient_hours if patient_hours > 0 else 0
        patient_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"  --- PATIENT {patient_id} REPORT ---")
        print(f"  Total Monitored Time: {patient_hours:.2f} Hours")
        print(f"  Sensitivity (Recall): {patient_sensitivity*100:.2f}%")
        print(f"  False Alarms: {fp} total -> FAR: {patient_far:.2f} False Positives / Hour")
        
        # Add to Global Trackers
        global_y_true.extend(y_true)
        global_y_pred.extend(y_pred)
        global_total_epochs += len(X_test)

    # ==========================================
    # GLOBAL GENERALIZATION EVALUATION
    # ==========================================
    print("\n====================================================")
    print("      GLOBAL GENERALIZATION REPORT (UNSEEN COHORT)    ")
    print("====================================================")
    
    tn_g, fp_g, fn_g, tp_g = confusion_matrix(global_y_true, global_y_pred).ravel()
    total_hours = global_total_epochs * epoch_duration / 3600.0
    global_far = fp_g / total_hours
    global_sensitivity = tp_g / (tp_g + fn_g)
    global_specificity = tn_g / (tn_g + fp_g)
    global_auc = roc_auc_score(global_y_true, global_y_pred)
    
    print(f"Total Cohort Monitored Time: {total_hours:.2f} Hours")
    print("-" * 50)
    print(f"TRUE POSITIVES (Seizures Caught): {tp_g}")
    print(f"FALSE NEGATIVES (Seizures Missed): {fn_g}")
    print(f"TRUE NEGATIVES (Normal Ignored): {tn_g}")
    print(f"FALSE POSITIVES (False Alarms): {fp_g}")
    print("-" * 50)
    print(f"GLOBAL SENSITIVITY (Recall) : {global_sensitivity*100:.2f}%")
    print(f"GLOBAL SPECIFICITY          : {global_specificity*100:.2f}%")
    print(f"GLOBAL AUC                  : {global_auc:.4f}")
    print(f"GLOBAL FALSE ALARM RATE (FAR): {global_far:.3f} per Hour")
    print("====================================================\n")
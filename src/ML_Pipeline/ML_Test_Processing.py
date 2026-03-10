"""
=============================================================================
EEG DATA PREPROCESSING & FEATURE EXTRACTION (CONTINUOUS TESTING SET)
=============================================================================
Description:
This script processes raw .edf EEG files exclusively for the unseen holdout 
group (Patients 21-24). Unlike the training script, it DOES NOT balance the 
dataset or shuffle the rows. 

It extracts 100% of the 2-second epochs in perfect chronological order to 
simulate a live hospital monitor. It tags every single row with its source 
Filename (e.g., 'chb21_21.edf') so the downstream Inference script can apply 
Temporal Median Filters without accidentally bleeding across different files.
Features are still scaled (Z-score) on a strict per-patient basis.
=============================================================================
"""

import mne
import os
import re
import numpy as np
import pandas as pd
from vmdpy import VMD
import EntropyHub as EH
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler


# 1. PARSER FUNCTION
def get_seizure_windows(summary_file_path, target_edf_name):
    """ Reads the CHB-MIT summary text files to extract seizure start/end timestamps. """
    seizure_windows = []
    if not os.path.exists(summary_file_path): 
        return seizure_windows
    
    with open(summary_file_path, 'r') as file: 
        lines = file.readlines()
        
    found_target_file = False
    num_seizures = 0
    
    for i, line in enumerate(lines):
        if line.startswith("File Name:") and target_edf_name in line:
            found_target_file = True
            continue 
            
        if found_target_file:
            if line.startswith("File Name:"): 
                break
            if line.startswith("Number of Seizures in File:"):
                num_seizures = int(re.findall(r'\d+', line)[-1])
                if num_seizures == 0: 
                    break 
            elif num_seizures > 0 and line.startswith("Seizure") and "Start Time" in line:
                start_sec = int(re.findall(r'\d+', line)[-1])
                end_line = lines[i+1]
                end_sec = int(re.findall(r'\d+', end_line)[-1])
                seizure_windows.append((start_sec, end_sec))
                
    return seizure_windows


# 2. MULTIPROCESSING WORKER FUNCTION
def process_single_epoch(epoch_idx, epoch_data, epoch_start, epoch_end, seizures, K, alpha, tau, DC, init, tol):
    """ Runs VMD and Entropy calculations on a single 2-second slice of EEG data. """
    row_features = []
    
    # Label the epoch (1 if it falls inside a seizure window, 0 otherwise)
    label = 0
    for (start_sec, end_sec) in seizures:
        if epoch_end > start_sec and epoch_start < end_sec:
            label = 1
            break
            
    # Extract Channels & Modes
    num_channels = epoch_data.shape[0]
    for ch_idx in range(num_channels):
        signal = epoch_data[ch_idx, :]
        u = VMD(signal, alpha, tau, K, DC, init, tol)
        
        for mode_idx in range(K):
            imf = u[mode_idx]
            tolerance = 0.2 * np.std(imf)
            
            # Safety Net: Prevent Division by Zero on completely flatline IMFs
            if tolerance < 1e-8:
                samp_en, fuzz_en, perm_en = 0.0, 0.0, 0.0
            else:
                samp_en = EH.SampEn(imf, m=2, r=tolerance)[0][-1]
                fuzz_en = EH.FuzzEn(imf, m=2, r=(tolerance, 2))[0][-1]
                perm_en = EH.PermEn(imf, m=3)[0][-1]
            
            row_features.extend([samp_en, fuzz_en, perm_en])
            
    # Append the ground-truth label to the very end of the feature row
    row_features.append(label)
    return row_features


# 3. MAIN EXTRACTION PIPELINE
if __name__ == "__main__":
    
    # Global Parameters
    epoch_duration = 2.0
    alpha, tau, K, DC, init, tol = 2000, 0.0, 8, 0, 1, 1e-7
    powerline_frequency = [60.0]

    skipped_files_list = []
    all_extracted_features = [] 
    

    # TARGET PATIENTS 21 TO 24 ONLY
    for patient_id_value in range(21, 25): 
        patient_id = f"{patient_id_value:02d}" 
        patient_folder = f"chbmit_dataset/chb{patient_id}"
        summary_file = os.path.join(patient_folder, f"chb{patient_id}-summary.txt")

        if not os.path.exists(patient_folder):
            print(f"Folder {patient_folder} not found, skipping...")
            continue

        edf_files = sorted([f for f in os.listdir(patient_folder) if f.endswith('.edf')])
        patient_features = [] 

        for edf_name in edf_files:
            file_path = os.path.join(patient_folder, edf_name)
            
            seizures = get_seizure_windows(summary_file, edf_name)

            print(f"\n================= PROCESSING {patient_id} | {edf_name} =================")
            
            # Load and Filter
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose='ERROR')
            raw.notch_filter(freqs=powerline_frequency, verbose='ERROR')
            raw.filter(l_freq=1.0, h_freq=40.0, verbose='ERROR')
            
            # Channel Mapping & Strict 8-Channel Selection
            channel_mapping = {ch: ch.replace('EEG ', '').replace(' ', '').upper() for ch in raw.ch_names}
            raw.rename_channels(channel_mapping)
            target_bases = ['FP1-F7', 'T7-P7', 'FP2-F8', 'T8-P8', 'F3-C3', 'C3-P3', 'F4-C4', 'C4-P4']
            actual_channels_to_keep = []
            
            for base in target_bases:
                matched_ch = next((ch for ch in raw.ch_names if base in ch), None)
                if matched_ch: actual_channels_to_keep.append(matched_ch)
            
            if len(actual_channels_to_keep) == 8:
                raw.pick_channels(actual_channels_to_keep, verbose='ERROR')
            else:
                print(f"  [!] Skipping {edf_name} due to missing channels.")
                skipped_files_list.append(f"Patient: {patient_id} File: {edf_name}")
                continue 
            
            # Downsample and Epoch
            raw.resample(128.0, verbose='ERROR')
            epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True, verbose='ERROR')
            data_matrix = epochs.get_data()
            num_epochs = data_matrix.shape[0]

            # KEEP 100% OF EPOCHS (NO INDEXING)
            print(f"  -> Extracting all {num_epochs} continuous epochs...")
            
            epoch_results = Parallel(n_jobs=-1, verbose=10)(
                delayed(process_single_epoch)(
                    idx, 
                    data_matrix[idx, :, :], 
                    idx * epoch_duration, 
                    (idx * epoch_duration) + epoch_duration, 
                    seizures, K, alpha, tau, DC, init, tol
                ) for idx in range(num_epochs)
            )
            
            # ADD FILENAME TO ROW
            for row in epoch_results:
                row.insert(0, edf_name) # Puts filename in the very first column
            
            patient_features.extend(epoch_results)

        # Apply Per-Patient Z-Score Normalization
        if patient_features:
            print(f"\n  -> Applying StandardScaler to Patient {patient_id} Baseline...")
            df_patient = pd.DataFrame(patient_features)
            
            # Isolate the math columns (Skip column 0 [Filename] and last column [Label])
            X_patient = df_patient.iloc[:, 1:-1].values 
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_patient)
            
            # Stitch the scaled data back into the dataframe
            df_patient.iloc[:, 1:-1] = X_scaled
            
            all_extracted_features.extend(df_patient.values.tolist())


# 4. FINAL SAVING (NO SHUFFLING OR BALANCING)
    print("\n====================================================")
    print("      TEST DATA EXTRACTION COMPLETE. SAVING CSV       ")
    print("====================================================")
    
    df_final = pd.DataFrame(all_extracted_features)
    
    col_names = ['Filename'] + [f'Feature_{i}' for i in range(df_final.shape[1]-2)] + ['Label']
    df_final.columns = col_names
    
    df_final.to_csv("generalised_testing_continuous.csv", index=False)
    print(f" CHRONOLOGICAL TEST DATASET SAVED! Total Continuous Rows: {len(df_final)}")

    if len(skipped_files_list) > 0:
        print(f"\nDATA LOSS REPORT: Skipped {len(skipped_files_list)} files due to missing channels:")
        for skipped_file in skipped_files_list:
            print(f"    - {skipped_file}")
    else:
        print("\n0 files skipped.")
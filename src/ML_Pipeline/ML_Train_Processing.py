"""
=============================================================================
EEG DATA PREPROCESSING & FEATURE EXTRACTION (TRAINING SET)
=============================================================================
Description:
This script processes raw .edf EEG files for Patients 1-20 to build a 
balanced, generalised training dataset. It applies powerline and bandpass 
filters, segments the data into 2-second epochs, and enforces a strict 
1:3 (Seizure : Normal) class ratio to prevent the Accuracy Paradox.

For each epoch, it targets 8 specific regional channels, decomposes the 
signals into 8 Intrinsic Mode Functions (IMFs) using Variational Mode 
Decomposition (VMD), and calculates Sample, Fuzzy, and Permutation Entropy. 
Crucially, the data is scaled (Z-score) on a strictly per-patient basis to 
erase baseline physiological differences before being stitched into the 
final master CSV.
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
        u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
        
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
    
    # Loop over the designated Training Patients (Patients 1-20 only)
    for patient_id_value in range(1, 21): 
        patient_id = f"{patient_id_value:02d}" 
        patient_folder = f"chbmit_dataset/chb{patient_id}"
        summary_file = os.path.join(patient_folder, f"chb{patient_id}-summary.txt")

        if not os.path.exists(patient_folder):
            print(f"Folder {patient_folder} not found, skipping...")
            continue

        edf_files = sorted([f for f in os.listdir(patient_folder) if f.endswith('.edf')])
        patient_features = [] # Temporary list for per-patient scaling

        for edf_name in edf_files:
            file_path = os.path.join(patient_folder, edf_name)
            print(f"\n================= PROCESSING {patient_id} | {edf_name} =================")
            
            seizures = get_seizure_windows(summary_file, edf_name)
            
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

            # Pre-Filter Indexing - avoid processing useless baseline data
            seizure_indices = []
            normal_indices = []
            
            for idx in range(num_epochs):
                ep_start = idx * epoch_duration
                ep_end = ep_start + epoch_duration
                is_seizure = False
                for (s_start, s_end) in seizures:
                    if ep_end > s_start and ep_start < s_end:
                        is_seizure = True
                        break
                
                if is_seizure: seizure_indices.append(idx)
                else: normal_indices.append(idx)

            # Keep all seizures, sample max 30 normal epochs per file
            num_normal_to_keep = min(30, len(normal_indices))
            if num_normal_to_keep > 0:
                sampled_normal_indices = np.random.choice(normal_indices, num_normal_to_keep, replace=False).tolist()
            else:
                sampled_normal_indices = []

            indices_to_process = seizure_indices + sampled_normal_indices
            
            if not indices_to_process:
                continue

            print(f"  -> Processing {len(seizure_indices)} Seizure & {len(sampled_normal_indices)} Baseline epochs")
            
            # Distribute workload across CPU cores
            epoch_results = Parallel(n_jobs=-1, verbose=10)(
                delayed(process_single_epoch)(
                    idx, 
                    data_matrix[idx, :, :], 
                    idx * epoch_duration, 
                    (idx * epoch_duration) + epoch_duration, 
                    seizures, K, alpha, tau, DC, init, tol
                ) for idx in indices_to_process
            )
            
            patient_features.extend(epoch_results)

        # Apply Per-Patient Z-Score Normalization
        if patient_features:
            print(f"\n  -> Applying StandardScaler to Patient {patient_id} Baseline...")
            patient_data_np = np.array(patient_features)
            
            X_patient = patient_data_np[:, :-1]  
            y_patient = patient_data_np[:, -1:]  
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_patient)
            
            scaled_patient_data = np.hstack((X_scaled, y_patient))
            all_extracted_features.extend(scaled_patient_data.tolist())


# 4. RATIO BALANCING & SAVING
    print("\n====================================================")
    print("      DATA EXTRACTION COMPLETE. BALANCING DATASET     ")
    print("====================================================")
    
    df_all = pd.DataFrame(all_extracted_features)
    
    label_col = df_all.columns[-1]
    df_seizures = df_all[df_all[label_col] == 1]
    df_normal_pool = df_all[df_all[label_col] == 0]
    
    total_seizures = len(df_seizures)
    target_normal_count = total_seizures * 3 
    
    print(f"Total Seizure Epochs Found: {total_seizures}")
    print(f"Targeting {target_normal_count} Normal Epochs for a 1:3 Ratio")
    
    if len(df_normal_pool) > target_normal_count:
        df_normal_final = df_normal_pool.sample(n=target_normal_count, random_state=42)
    else:
        print("[!] Warning: Not enough normal epochs collected to reach 1:3 ratio. Using all available.")
        df_normal_final = df_normal_pool
        
    # Shuffle and Save
    df_final = pd.concat([df_seizures, df_normal_final]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_final.to_csv("generalised_training_subset.csv", index=False)
    print(f"MASTER DATASET SAVED! Total Rows: {len(df_final)}")

    if len(skipped_files_list) > 0:
        print(f"\nDATA LOSS REPORT: Skipped {len(skipped_files_list)} files due to missing channels:")
        for skipped_file in skipped_files_list:
            print(f"    - {skipped_file}")
    else:
        print("\n0 files skipped.")

"""
EXPECTED TERMINAL OUTPUT REFERENCE:
====================================================
      DATA EXTRACTION COMPLETE. BALANCING DATASET     
====================================================
Total Seizure Epochs Found: 4970
Targeting 14910 Normal Epochs for a 1:3 Ratio...
MASTER DATASET SAVED! Total Rows: 19880

[!] DATA LOSS REPORT: Skipped 3 files due to missing channels:
    - Patient: 12 File: chb12_27.edf
    - Patient: 12 File: chb12_28.edf
    - Patient: 12 File: chb12_29.edf
"""
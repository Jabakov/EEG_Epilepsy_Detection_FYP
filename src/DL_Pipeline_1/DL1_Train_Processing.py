"""
=============================================================================
DEEP LEARNING PREPROCESSING (TRAINING SET)
=============================================================================
Description:
This script prepares raw EEG data for Deep Learning models (CNNs, LSTMs).
It targets Patients 1-20, applying Bandpass (1-40Hz) and Notch (60Hz) filters.
It strictly enforces an 18-channel Longitudinal Bipolar Montage.

Crucially, it skips manual feature extraction. Instead, it slices the raw 
EEG into 2-second non-overlapping epochs (18 channels x 256 timesteps). 
It balances the classes (All Seizures + 30 Normal epochs per file) and exports 
the data as a highly efficient compressed 3D NumPy Tensor (.npz) file, 
ready for direct injection into PyTorch or TensorFlow DataLoaders.
=============================================================================
"""

import mne
import os
import re
import numpy as np
from joblib import Parallel, delayed

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


# 2. MAIN EXTRACTION PIPELINE
if __name__ == "__main__":
    
    # Global Parameters
    epoch_duration = 2.0
    powerline_frequency = [60.0]
    sampling_rate = 128.0

    skipped_files_list = []
    
    # Master lists to hold the raw 3D grids and their labels
    X_master = []
    Y_master = []

    # TARGET PATIENTS 1 TO 20 (Note: I renamed the file logic to DL_Train to reflect Patients 1-20)
    for patient_id_value in range(1, 21): 
        patient_id = f"{patient_id_value:02d}" 
        patient_folder = f"chbmit_dataset/chb{patient_id}"
        summary_file = os.path.join(patient_folder, f"chb{patient_id}-summary.txt")

        if not os.path.exists(patient_folder):
            continue

        edf_files = sorted([f for f in os.listdir(patient_folder) if f.endswith('.edf')])

        for edf_name in edf_files:
            file_path = os.path.join(patient_folder, edf_name)
            seizures = get_seizure_windows(summary_file, edf_name)

            print(f"\n================= PROCESSING {patient_id} | {edf_name} =================")
            
            # Load and Filter
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose='ERROR')
            raw.notch_filter(freqs=powerline_frequency, verbose='ERROR')
            raw.filter(l_freq=1.0, h_freq=40.0, verbose='ERROR')
            
            # Channel Mapping: Clean up standard names
            channel_mapping = {ch: ch.replace('EEG ', '').replace(' ', '').upper() for ch in raw.ch_names}
            raw.rename_channels(channel_mapping)
            
            # THE 8-CHANNEL "EDGE-OPTIMISED" MONTAGE
            target_bases = [
                'FP1-F7', 'T7-P7',  # Left Temporal
                'FP2-F8', 'T8-P8',  # Right Temporal
                'F3-C3', 'C3-P3',   # Left Parasagittal
                'F4-C4', 'C4-P4'    # Right Parasagittal
            ]
            
            actual_channels_to_keep = []
            for base in target_bases:
                matched_ch = next((ch for ch in raw.ch_names if base in ch), None)
                if matched_ch: actual_channels_to_keep.append(matched_ch)
            
            # Bouncer: Must have exactly 8 channels
            if len(actual_channels_to_keep) == 8:
                raw.pick_channels(actual_channels_to_keep, verbose='ERROR')
                raw.reorder_channels(actual_channels_to_keep)
            else:
                print(f"  [!] Skipping {edf_name} due to missing/incompatible channels.")
                skipped_files_list.append(f"Patient: {patient_id} File: {edf_name}")
                continue 
            
            raw.resample(sampling_rate, verbose='ERROR')
            data_matrix = raw.get_data() # Shape: (8, Total_Samples)
            
            # THE DYNAMIC SLIDING WINDOW (Pipeline 2 Core Logic)
            window_size = int(epoch_duration * sampling_rate)  # 256 samples
            normal_step = int(2.0 * sampling_rate)             # 256 samples (0% overlap)
            seizure_step = int(0.25 * sampling_rate)           # 32 samples (87.5% overlap)
            
            # Convert seizure times from seconds to samples
            seizure_samples = [(start * sampling_rate, end * sampling_rate) for start, end in seizures]
            
            current_start = 0
            total_samples = data_matrix.shape[1]
            
            extracted_seizures = 0
            extracted_normals = 0

            while current_start + window_size <= total_samples:
                current_end = current_start + window_size
                
                # Check if this specific window touches a seizure
                is_seizure = False
                for (s_start, s_end) in seizure_samples:
                    if current_end > s_start and current_start < s_end:
                        is_seizure = True
                        break
                
                if is_seizure:
                    X_master.append(data_matrix[:, current_start:current_end])
                    Y_master.append(1)
                    current_start += seizure_step # Slide slowly (Overlap)
                    extracted_seizures += 1
                else:
                    X_master.append(data_matrix[:, current_start:current_end])
                    Y_master.append(0)
                    current_start += normal_step # Slide normally (No Overlap)
                    extracted_normals += 1
                    
            print(f"  -> Extracted {extracted_seizures} Overlapping Seizure & {extracted_normals} Baseline tensors.")


# 3. CLASS BALANCING & TENSOR EXPORT
    print("\n====================================================")
    print("      DEEP LEARNING DATA EXTRACTION COMPLETE          ")
    print("====================================================")
    
    # Separate the lists to calculate the 3:1 ratio
    X_seizures = [X_master[i] for i in range(len(Y_master)) if Y_master[i] == 1]
    X_normals = [X_master[i] for i in range(len(Y_master)) if Y_master[i] == 0]
    
    num_seizures = len(X_seizures)
    target_normals = num_seizures * 3  # The 3:1 Ratio!
    
    # Downsample the normal data
    if len(X_normals) > target_normals:
        indices = np.random.choice(len(X_normals), target_normals, replace=False)
        X_normals_balanced = [X_normals[i] for i in indices]
    else:
        X_normals_balanced = X_normals # Just in case we have very few normals
        
    print(f"\nBalancing Complete: {len(X_seizures)} Seizures vs {len(X_normals_balanced)} Normals (3:1 Ratio)")

    # Recombine and convert to final massive NumPy arrays
    X_final = np.array(X_seizures + X_normals_balanced)
    Y_final = np.array([1]*len(X_seizures) + [0]*len(X_normals_balanced))
    
    # Save the pipeline 2 dataset
    export_filename = "dl_training_tensors_pipeline2.npz"
    np.savez_compressed(export_filename, X=X_final, y=Y_final)
    print(f"PIPELINE 2 DATASET SAVED AS '{export_filename}'")
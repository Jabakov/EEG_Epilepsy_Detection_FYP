import mne
import os
import re
import numpy as np
import pandas as pd
from vmdpy import VMD
import EntropyHub as EH
from joblib import Parallel, delayed

# ==========================================
# 1. PARSER FUNCTION
# ==========================================
def get_seizure_windows(summary_file_path, target_edf_name):
    seizure_windows = []
    if not os.path.exists(summary_file_path): return seizure_windows
    with open(summary_file_path, 'r') as file: lines = file.readlines()
        
    found_target_file = False
    num_seizures = 0
    for i, line in enumerate(lines):
        if line.startswith("File Name:") and target_edf_name in line:
            found_target_file = True
            continue 
        if found_target_file:
            if line.startswith("File Name:"): break
            if line.startswith("Number of Seizures in File:"):
                num_seizures = int(re.findall(r'\d+', line)[-1])
                if num_seizures == 0: break 
            elif num_seizures > 0 and line.startswith("Seizure") and "Start Time" in line:
                start_sec = int(re.findall(r'\d+', line)[-1])
                end_line = lines[i+1]
                end_sec = int(re.findall(r'\d+', end_line)[-1])
                seizure_windows.append((start_sec, end_sec))
    return seizure_windows

# ==========================================
# 2. MULTIPROCESSING WORKER FUNCTION
# ==========================================
# This function handles exactly ONE epoch. Joblib will run this on 8-16 cores simultaneously!
def process_single_epoch(epoch_idx, epoch_data, epoch_start, epoch_end, seizures, K, alpha, tau, DC, init, tol):
    row_features = []
    
    # 1. Labeling
    label = 0
    for (start_sec, end_sec) in seizures:
        if epoch_end > start_sec and epoch_start < end_sec:
            label = 1
            break
            
    # 2. Extract Channels & Modes
    num_channels = epoch_data.shape[0]
    for ch_idx in range(num_channels):
        signal = epoch_data[ch_idx, :]
        u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
        
        for mode_idx in range(K):
            imf = u[mode_idx]
            tolerance = 0.2 * np.std(imf)
            
            # SAFETY NET: Prevent Division by Zero on flatline IMFs
            if tolerance < 1e-8:
                samp_en, fuzz_en, perm_en = 0.0, 0.0, 0.0
            else:
                samp_en = EH.SampEn(imf, m=2, r=tolerance)[0][-1]
                fuzz_en = EH.FuzzEn(imf, m=2, r=(tolerance, 2))[0][-1]
                perm_en = EH.PermEn(imf, m=3)[0][-1]
            
            row_features.extend([samp_en, fuzz_en, perm_en])
            
    # 3. Append Label to the very end
    row_features.append(label)
    return row_features

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    epoch_duration = 2.0
    alpha, tau, K, DC, init, tol = 2000, 0.0, 8, 0, 1, 1e-7
    powerline_frequency = [60.0]

    all_extracted_features = []
    for patient_id_value in range(1, 25):
        patient_id = f"{patient_id_value:02d}" 
        patient_folder = f"chbmit_dataset/chb{patient_id}"
        summary_file = os.path.join(patient_folder, f"chb{patient_id}-summary.txt")

        # FIXED: Wrapped in sorted() so it starts at 01, not 18!
        edf_files = sorted([f for f in os.listdir(patient_folder) if f.endswith('.edf')])

        for edf_name in edf_files:
            file_path = os.path.join(patient_folder, edf_name)
            print(f"\n================= PROCESSING {edf_name} =================")
            
            seizures = get_seizure_windows(summary_file, edf_name)
            if seizures: print(f"Found Seizures at: {seizures}")
            else: print("Normal Baseline File (No Seizures)")

            raw = mne.io.read_raw_edf(file_path, preload=True, verbose='ERROR')
            raw.notch_filter(freqs=powerline_frequency, verbose='ERROR')
            raw.filter(l_freq=1.0, h_freq=40.0, verbose='ERROR')
            
            # --- THE CHANNEL CLEANUP ---
            print("  -> Removing dummy and non-EEG channels...")
            
            # List of known garbage channels in CHB-MIT
            bad_channels = ['-', 'ECG', 'VNS']
            
            # Find which of those bad channels actually exist in this specific file
            channels_to_drop = [ch for ch in bad_channels if ch in raw.ch_names]
            
            if channels_to_drop:
                raw.drop_channels(channels_to_drop)
                print(f"Dropped: {channels_to_drop}")

            # THE SPEED HACK: Downsample to 128 Hz before epoching!
            print("  -> Downsampling to 128 Hz...")
            raw.resample(128.0, verbose='ERROR')

            epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True, verbose='ERROR')
            data_matrix = epochs.get_data()
            
            print(f"  -> Extracting features across all CPU cores...")
            
            # THE MULTIPROCESSING HACK: n_jobs=-1 tells Python to use 100% of your CPU cores
            # verbose=10 gives you a beautiful progress bar in your terminal!
            epoch_results = Parallel(n_jobs=-1, verbose=10)(
                delayed(process_single_epoch)(
                    epoch_idx, 
                    data_matrix[epoch_idx, :, :], # Pass only this epoch's data to save RAM
                    epoch_idx * epoch_duration, 
                    (epoch_idx * epoch_duration) + epoch_duration, 
                    seizures, K, alpha, tau, DC, init, tol
                ) for epoch_idx in range(data_matrix.shape[0])
            )
            
            # epoch_results is a list of rows. We extend our master list with them.
            all_extracted_features.extend(epoch_results)

        print("\nSaving final dataset to CSV...")
        df = pd.DataFrame(all_extracted_features)
        df.to_csv(f"chb{patient_id}_extracted_features.csv", index=False)
        print("Pipeline Complete!")
        
        # Without using all cores one file takes 3 hours / 1 epoch per 7 seconds
        # With all cores plus downsampling to 128 Hz, one file takes 5 minutes, this would still take 55 hours for all data
        # Removed redundant signals such as ECG, VNS, and dummy channels, which were causing errors in VMD and entropy calculations. This also speeds up the process by reducing the number of channels we analyze.
        # Moved from 24 Channels to 8 channels, one edf file took 2mins to finish
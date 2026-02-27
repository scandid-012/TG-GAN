dataset_path = '../input/bci-competition-iv-dataset-2b/' # Standard Kaggle input path
eeg_channels_of_interest = ['EEG:C3', 'EEG:Cz', 'EEG:C4']
eog_channels = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03'] # Good to define to set their types correctly

import os
import glob
import mne
import numpy as np
import pandas as pd # Optional, for managing metadata
import traceback # For detailed error reporting

print(f"MNE version: {mne.__version__}")

dataset_path = '../input/bci-competition-iv-dataset-2b/'
eeg_channels_of_interest = ['EEG:C3', 'EEG:Cz', 'EEG:C4']
eog_channels_for_typing = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']

def extract_eeg_from_gdf(file_path, target_eeg_names, target_eog_names):
    """
    Loads a GDF file, sets channel types, picks specified EEG channels, and returns their data.
    """
    print(f"\n--- Processing file: {os.path.basename(file_path)} ---")
    try:
        # 1. Load GDF data
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose='WARNING')
        all_chs_in_file = raw.ch_names
        print(f"File {os.path.basename(file_path)}: Loaded. Channels found in file: {all_chs_in_file}. Initial types: {raw.get_channel_types(unique=True)}")

        # 2. Identify which of our target channels are actually in this file
        actual_eeg_chs_found_by_name = [ch for ch in target_eeg_names if ch in all_chs_in_file]
        actual_eog_chs_found_by_name = [ch for ch in target_eog_names if ch in all_chs_in_file]

        print(f"File {os.path.basename(file_path)}: Target EEG names found: {actual_eeg_chs_found_by_name}")
        print(f"File {os.path.basename(file_path)}: Target EOG names found: {actual_eog_chs_found_by_name}")

        # 3. Build the mapping to set channel types correctly
        type_mapping = {}
        for ch_name in actual_eeg_chs_found_by_name:
            type_mapping[ch_name] = 'eeg'
        for ch_name in actual_eog_chs_found_by_name:
            type_mapping[ch_name] = 'eog'

        if not type_mapping:
            print(f"File {os.path.basename(file_path)}: WARNING - None of the target EEG/EOG channels ({target_eeg_names} / {target_eog_names}) found by name. Cannot reliably set types or pick EEG data by specified names. Skipping this file.")
            return None, None
        
        raw.set_channel_types(type_mapping)
        print(f"File {os.path.basename(file_path)}: Applied type mapping: {type_mapping}. Current unique types: {raw.get_channel_types(unique=True)}")

        # 4. Pick only the channels now designated as 'eeg' (using their names)
        # actual_eeg_chs_found_by_name is already guaranteed to contain only channels present in raw.
        raw_eeg_picked = raw.copy().pick(actual_eeg_chs_found_by_name) # REMOVED on_missing='ignore'

        if not raw_eeg_picked.ch_names:
            print(f"File {os.path.basename(file_path)}: ERROR - No channels were picked from the list {actual_eeg_chs_found_by_name} after type setting. This is unexpected. Skipping file.")
            return None, None

        # Final check on picked channels' types
        picked_types = raw_eeg_picked.get_channel_types()
        if not all(ch_type == 'eeg' for ch_type in picked_types):
            print(f"File {os.path.basename(file_path)}: WARNING - Picked channels ({raw_eeg_picked.ch_names}) include non-EEG types: {picked_types} after picking by name. Original mapping was {type_mapping}. This could indicate an issue. Skipping file.")
            return None, None
            
        eeg_data = raw_eeg_picked.get_data()
        sfreq = raw_eeg_picked.info['sfreq']
        
        print(f"File {os.path.basename(file_path)}: SUCCESS - Extracted {eeg_data.shape[0]} EEG channels ({raw_eeg_picked.ch_names}), {eeg_data.shape[1]} samples, sfreq: {sfreq} Hz.")
        return eeg_data, sfreq
        
    except Exception as e:
        print(f"File {os.path.basename(file_path)}: CRITICAL ERROR - {e}")
        traceback.print_exc()
        return None, None

# --- Main loop (should be the same as before) ---
all_gdf_files = sorted(glob.glob(os.path.join(dataset_path, '*.gdf')))
all_eeg_data_list = []
file_metadata = [] # To store metadata like filename, original shape, etc.

print(f"Found {len(all_gdf_files)} GDF files to process.")

for gdf_file_path in all_gdf_files:
    # Use the corrected function here
    eeg_data, sfreq = extract_eeg_from_gdf(gdf_file_path, eeg_channels_of_interest, eog_channels_for_typing)
    
    if eeg_data is not None:
        all_eeg_data_list.append(eeg_data)
        # For metadata, it's safer to get channel names from the successfully picked object if possible
        # However, `raw_eeg_picked.ch_names` is not directly returned by the function.
        # For now, we'll assume the number of channels in eeg_data.shape[0] corresponds to our target channels.
        file_metadata.append({
            'filename': os.path.basename(gdf_file_path),
            'n_eeg_channels_extracted': eeg_data.shape[0],
            'n_timepoints': eeg_data.shape[1],
            'sampling_frequency': sfreq
            # 'extracted_channel_names': # This would be ideal to get from raw_eeg_picked.ch_names
        })
    else:
        print(f"File {os.path.basename(gdf_file_path)}: SKIPPED due to errors or inability to reliably extract target EEG channels.")

print(f"\n--- Processing Complete ---")
print(f"Successfully extracted EEG data from {len(all_eeg_data_list)} out of {len(all_gdf_files)} files.")

if all_eeg_data_list:
    metadata_df = pd.DataFrame(file_metadata)
    print("\nMetadata of extracted files (first 5):")
    print(metadata_df.head())
    if len(all_eeg_data_list) > 0 and all_eeg_data_list[0] is not None:
     print(f"\nShape of EEG data from the first successfully extracted file: {all_eeg_data_list[0].shape}")
else:
    print("No EEG data was extracted from any file.")
